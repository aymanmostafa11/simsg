"""
    This file should run SIMSG's API
    it is basically a copy of SIMSG_gui.py but with removing all UI and data loader related code,
    It should only contain code for:

        1. Get input image, original graph, modified graph from client
        2. Preprocess input to be given to the model
        3. Run the model
        4. Return generated image to client

"""

import argparse
import json
import os
from PIL import Image
from builtins import enumerate

import numpy as np
import torch

from simsg.data import imagenet_deprocess_batch
from simsg.model import SIMSGModel
from simsg.utils import int_tuple, bool_flag
from flask import Flask, make_response

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='./experiments/vg/spade_64_vg_model.pt')
parser.add_argument('--dataset', default='vg', choices=['clevr', 'vg'])
parser.add_argument('--data_h5', default=None)
parser.add_argument('--predgraphs', default=False, type=bool_flag)
parser.add_argument('--image_size', default=(128, 128), type=int_tuple)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--update_input', default=True, type=bool_flag)
parser.add_argument('--shuffle', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=1, type=int)

# deterministic vs diverse results
# instead of having zeros as visual feature, choose a random one from our feature distribution
parser.add_argument('--random_feats', default=False, type=bool_flag)

args = parser.parse_args()
args.mode = "eval"
if args.dataset == "clevr":
    assert args.random_feats == False
    DATA_DIR = "./datasets/clevr/target/"
    args.data_image_dir = DATA_DIR
else:
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


app = Flask(__name__)


class API():
    def __init__(self):
        self.model = build_model()
        self.mode = "auto_withfeats"
        self.new_objs = None
        self.new_triples = None


    """
        Ideally, this is how the end point should look like.
        
        ASSUMPTIONS: 
            1. preprocess_image and generate_image work (wow)
            2. self.image is a PIL image
    """
    @app.route('/generate', methods=['POST'])
    def generate(self):
        self.preprocess_image()
        self.generate_image()

        image_bytes = self.image.tobytes()

        response = make_response(image_bytes)
        response.headers.set('Content-Type', 'image/png')
        response.headers.set('Content-Disposition', 'attachment', filename='kaka.png')
        return response

    """
        This is a copy of the function getfile() in SIMSG_gui.py without UI related stuff
        it should extract all the inputs that the model requires to run
    """
    # TODO: Modify it to use the image sent by the client
    def preprocess_image(self):
        """
        Loads input data
        """
        global vocab

        available_tst = ["sheep", "man_on_horse", "rmdn"]
        img = available_tst[0]
        load_image = Image.open("./simsg/tmp/" + img + ".jpg")
        arr = np.expand_dims(np.array(load_image.resize((64, 64)), dtype=np.uint8).transpose((2, 0, 1)), axis=0)

        if img in ["sheep", "man_on_horse"]:
            vocab = json.load(open("./simsg/tmp/custom_data_info.json", 'r'))
        else:
            vocab = json.load(open("./simsg/tmp/rmdn_data_info.json", 'r'))

        vocab['object_idx_to_name'] = vocab['ind_to_classes']
        del vocab['ind_to_classes']

        vocab['pred_idx_to_name'] = vocab['ind_to_predicates']
        del vocab['ind_to_predicates']

        if img == "sheep":
            pred = json.load(open('./simsg/tmp/custom_prediction.json', 'r'))['0']
        elif img == "man_on_horse":
            pred = json.load(open('./simsg/tmp/custom_prediction.json', 'r'))['1']
        elif img == "rmdn":
            pred = json.load(open('./simsg/tmp/rmdn_prediction.json', 'r'))['0']

        topk_objs = 8
        topk_rels = 3
        """
        'rel_pair' contain a local indexer into the 'bbox_labels'
        """

        triples = []  # triplet containing (local obj label, predicate label, local subj label)
        objs = pred["bbox_labels"][:topk_objs]
        boxes = pred["bbox"][:topk_objs]
        for i, pair in enumerate(pred['rel_pairs']):
            obj = pair[0]
            predicate = pred['rel_labels'][i]
            subj = pair[1]
            if obj < topk_objs and subj < topk_objs:  # only take relations between topk objects
                triples.append([obj, predicate, subj])

        triples = triples[:topk_rels]

        self.imgs = torch.tensor(arr).cpu()
        self.objs = torch.tensor(objs).cpu()
        self.boxes = torch.tensor(boxes).cpu()
        self.triples = torch.tensor(triples).cpu()

        self.keep_box_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
        self.keep_feat_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
        self.keep_image_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
        self.combine_gt_pred_box_idx = torch.zeros_like(self.objs)
        self.added_objs_idx = torch.zeros_like(self.objs.unsqueeze(1), dtype=torch.float)

        self.new_triples, self.new_objs = None, None

        # image = imagenet_deprocess_batch(self.imgs)  # put image in range 0-255 instead of 0-1
        image = self.imgs[0].numpy().transpose(1, 2, 0).copy()  # change order to (w, h, c)

        self.image = image
        self.imgs_in = torch.tensor(np.append(arr, np.zeros([1, 1, arr.shape[-1], arr.shape[-2]]), axis=1),
                                    dtype=torch.float).cpu()


    """
        This is a copy of the function gen_img() in SIMSG_gui.py, it should generate a new image from the 
        preprocessed client inputs
    """
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


if __name__ == '__main__':
    api = API()
    app.run(debug=True)
