"""
    This file should run SIMSG's API
    it is basically a copy of SIMSG_gui.py but with removing all UI and data loader related code,
    It should only contain code for:

        1. Get input image, original graph, modified graph from client
        2. Preprocess input to be given to the model
        3. Run the model
        4. Return generated image to client

"""
from builtins import enumerate

import networkx as nx
from networkx.readwrite import json_graph

import os, json, argparse
from simsg.model import SIMSGModel
import torch
from simsg.data import imagenet_deprocess_batch
from simsg.loader_utils import build_eval_loader
from simsg.utils import int_tuple, bool_flag

import scripts.eval_utils as eval_utils
from flask import Flask, send_file
from PIL import Image

import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='./experiments/vg/spade_64_vg_model.pt')
parser.add_argument('--dataset', default='vg', choices=['clevr', 'vg'])
parser.add_argument('--data_h5', default=None)
parser.add_argument('--predgraphs', default=False, type=bool_flag)
parser.add_argument('--image_size', default=(256, 256), type=int_tuple)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--update_input', default=True, type=bool_flag)
parser.add_argument('--shuffle', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=1, type=int)
# deterministic vs diverse results
# instead of having zeros as visual feature, choose a random one from our feature distribution
parser.add_argument('--random_feats', default=False, type=bool_flag)

args = parser.parse_args()
args.mode = "eval"

DATA_DIR = "./datasets/vg/"
args.data_image_dir = os.path.join(DATA_DIR, 'images')

if args.data_h5 is None:
    if args.predgraphs:
        args.data_h5 = os.path.join(DATA_DIR, 'test_predgraphs.h5')
    else:
        args.data_h5 = os.path.join(DATA_DIR, 'test.h5')

vocab_json = os.path.join(DATA_DIR, "vocab.json")
with open(vocab_json, 'r') as f:
    vocab = json.load(f)

preds = sorted(vocab['pred_idx_to_name'])
objs = sorted(vocab['object_idx_to_name'])

checkpoint = None

IMG_DISPLAY_SIZE = 400


def remove_vgg(model_state):
    def filt(pair):
        key, val = pair
        return "high_level_feat" not in key

    return dict(filter(filt, model_state.items()))


def build_model():
    global checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))

    model = SIMSGModel(**checkpoint['model_kwargs'])
    new_state = remove_vgg(checkpoint['model_state'])
    model.load_state_dict(new_state, strict=False)
    model.eval()
    model.image_size = args.image_size
    #   model.cuda()
    return model




class Model():
    def __init__(self):
        self.model = build_model()
        self.data_loader = iter(build_eval_loader(args, checkpoint, no_gt=True))
        self.mode = "auto_withfeats"
        self.new_objs = None
        self.new_triples = None

    def get_image(self):
        self.load_vg_image()

        img = Image.fromarray(self.image, "RGB")
        return img


    """
        This is a copy of the function getfile() in SIMSG_gui.py without UI related stuff
        it should extract all the inputs that the model requires to run
    """
    def load_vg_image(self):
        """
        Loads input data
        """
        self.batch = next(self.data_loader)

        # self.imgs, self.objs, self.boxes, self.triples, self.obj_to_img, self.triple_to_img, self.imgs_in = \
        #     [x.cuda() for x in self.batch]
        self.imgs, self.objs, self.boxes, self.triples, self.obj_to_img, self.triple_to_img, self.imgs_in = \
            [x for x in self.batch]

        self.keep_box_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
        self.keep_feat_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
        self.keep_image_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
        self.combine_gt_pred_box_idx = torch.zeros_like(self.objs)
        self.added_objs_idx = torch.zeros_like(self.objs.unsqueeze(1), dtype=torch.float)

        self.new_triples, self.new_objs = None, None

        image = imagenet_deprocess_batch(self.imgs)
        image = image[0].numpy().transpose(1, 2, 0).copy()

        self.image = image


    def preprocess_graph(self, triples):
        """
        Prepares graphs in the right format for networkx
        """
        if self.new_objs is not None:
            objs = self.new_objs.cpu().numpy()
        else:
            objs = self.objs
        new_triples = []
        boxes = self.boxes.cpu().numpy()
        boxes_ = {}
        triple_idx = 0
        for [s, p, o] in triples:
            s2 = vocab['object_idx_to_name'][objs[s]] + "." + str(s)
            o2 = vocab['object_idx_to_name'][objs[o]] + "." + str(o)
            p2 = vocab['pred_idx_to_name'][p] + "." + str(triple_idx)
            new_triples.append([s2, p2, o2])

            x1_o, y1_o, x2_o, y2_o = boxes[o]
            x1_s, y1_s, x2_s, y2_s = boxes[s]
            xc_o = x1_o + (x2_o - x1_o) / 2
            yc_o = y1_o + (y2_o - y1_o) / 2
            xc_s = x1_s + (x2_s - x1_s) / 2
            yc_s = y1_s + (y2_s - y1_s) / 2
            x_p = (xc_o + xc_s) / 2
            y_p = (yc_o + yc_s) / 2
            if vocab['pred_idx_to_name'][p] in boxes_.keys():
                old_xc, old_yc = boxes_[vocab['pred_idx_to_name'][p]]
                boxes_[vocab['pred_idx_to_name'][p] + "." + str(triple_idx)] = [1 - ((x_p + old_xc) / 2),
                                                                                1 - ((y_p + old_yc) / 2)]
            else:
                boxes_[vocab['pred_idx_to_name'][p] + "." + str(triple_idx)] = [1 - x_p, 1 - y_p]

            triple_idx += 1

        for i, obj in enumerate(objs):
            x1, y1, x2, y2 = boxes[i]
            xc = x1 + (x2 - x1) / 2
            yc = y1 + (y2 - y1) / 2
            boxes_[vocab['object_idx_to_name'][obj] + "." + str(i)] = [1 - xc, 1 - yc]

        return new_triples, boxes_


    def get_graph(self):
        """
        Initializes new networkx graph from the current state of the objects and triples
        and draws the graph on canvas
        """
        self.graph = nx.DiGraph()
        if self.new_triples is not None:
            curr_triples = self.new_triples.cpu().numpy()
        else:
            curr_triples = self.triples.cpu().numpy()
        self.curr_triples, self.pos = self.preprocess_graph(curr_triples)

        i = 0
        import matplotlib.patches
        astyle = matplotlib.patches.ArrowStyle.Fancy(head_length=.4, head_width=.4, tail_width=.4)

        for s, p, o in self.curr_triples:
            self.graph.add_node(s)
            if "__image__" not in s and "__in_image__" not in p and "__image__" not in o:
                # make s->p edge thicker than p->o, to indicate direction
                self.graph.add_edge(s, p, width= 2, arrows=True, arrowstyle=astyle)
                self.graph.add_edge(p, o, width=1)
                i += 1


        for node in self.graph.nodes:
            self.graph.nodes[node]['color'] = 'w'
            self.graph.nodes[node]['edgecolor'] = 'g'
            self.graph.nodes[node]['size'] = 2500

            for edge_attribute in self.graph[node].values():
                edge_attribute['arrows'] = True

        # Remove the arrow style attribute from the edges
        # (Causes problems in serialization)
        for u, v, d in self.graph.edges(data=True):
            if "arrowstyle" in d:
                del d["arrowstyle"]

        print(self.graph)

        graph_as_json = json_graph.node_link_data(self.graph)

        return graph_as_json


    def generate_image(self):
        """
        Generates an image, as indicated by the modified graph
        """
        if self.new_triples is not None:
            triples_ = self.new_triples
        else:
            triples_ = self.triples

        query_feats = None

        model_out = self.model(self.new_objs, triples_, None,
                               boxes_gt=self.boxes, masks_gt=None, src_image=self.imgs_in, mode=self.mode,
                               query_feats=query_feats, keep_box_idx=self.keep_box_idx,
                               keep_feat_idx=self.keep_feat_idx, combine_gt_pred_box_idx=self.combine_gt_pred_box_idx,
                               keep_image_idx=self.keep_image_idx, random_feats=args.random_feats,
                               get_layout_boxes=True)

        imgs_pred, boxes_pred, masks_pred, noised_srcs, _, layout_boxes = model_out

        image = imagenet_deprocess_batch(imgs_pred)
        image = image[0].detach().numpy().transpose(1, 2, 0).copy()
        if args.update_input:
            self.image = image.copy()


        if args.update_input:
            # reset everything so that the predicted image is now the input image for the next step
            self.imgs = imgs_pred.detach().clone()
            self.imgs_in = torch.cat([self.imgs, torch.zeros_like(self.imgs[:,0:1,:,:])], 1)
            self.boxes = layout_boxes.detach().clone()
            self.keep_box_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
            self.keep_feat_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
            self.keep_image_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
            self.combine_gt_pred_box_idx = torch.zeros_like(self.objs)
        else:
            # input image is still the original one - don't reset anything
            # if an object is added for the first time, the GT/input box is still a dummy (set in add_triple)
            # in this case, we update the GT/input box, using the box predicted from SGN,
            # so that it can be used in future changes that rely on the GT/input box, e.g. replacement
            self.boxes = self.added_objs_idx * layout_boxes.detach().clone() + (1 - self.added_objs_idx) * self.boxes
            self.added_objs_idx = torch.zeros_like(self.objs.unsqueeze(1), dtype=torch.float)

    def remove_node(self, selected_node):
        """
        Removes an object node and all its connections
        Used in the object removal mode
        """
        if selected_node is not None:

            idx = selected_node
            # remove node and all connecting edges
            self.new_objs, self.new_triples, self.boxes, self.imgs_in, self.obj_to_img, _ = \
                        eval_utils.remove_node(self.objs, self.triples,
                                               self.boxes, self.imgs_in, [idx],
                                               torch.zeros_like(self.objs),
                                               torch.zeros_like(self.triples))

            # update keep arrays
            idlist = list(range(self.objs.shape[0]))
            keep_idx = [i for i in idlist if i != idx]
            self.keep_box_idx = self.keep_box_idx[keep_idx]
            self.keep_feat_idx = self.keep_feat_idx[keep_idx]
            self.keep_image_idx = self.keep_image_idx[keep_idx]
            self.added_objs_idx = self.added_objs_idx[keep_idx]
            self.combine_gt_pred_box_idx = self.combine_gt_pred_box_idx[keep_idx]

            self.objs = self.new_objs
            self.triples = self.new_triples

            # update the networkx graph for visualization
            self.mode = "remove"



app = Flask(__name__)
@app.route("/load_vg_data")
def get_vg_data():
    image = model.get_image()
    graph = model.get_graph()

    image.save('api_data/vg/vg_image.jpeg')

    with open("api_data/vg/vg_graph.json", "w") as outfile:
        json.dump(graph, outfile)

    archived = shutil.make_archive('vg_data', 'zip', 'api_data/vg/')

    return send_file(archived, mimetype='zip')


if __name__ == '__main__':
    global model
    model = Model()
    app.run(debug=True)

