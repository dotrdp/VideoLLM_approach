# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.

import io

import numpy as np
import PIL.Image
import torch
from IPython.display import Image, display

from transformers import AutoTokenizer, VisualBertForVisualReasoning

from importlib.machinery import SourceFileLoader
utils = SourceFileLoader("utils", "NOTCODEDBYME/utils.py").load_module()
visual_cues = SourceFileLoader("visual_cues", "NOTCODEDBYME/visual_cues.py").load_module()
conf = utils.Config
frcnn_cfg = conf.from_pretrained("unc-nlp/frcnn-vg-finetuned")

frcnn = visual_cues.GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

image_preprocess = utils.Preprocess(frcnn_cfg)





images, sizes, scales_yx = image_preprocess("TEST.jpg")
output_dict = frcnn(
    images,
    sizes,
    scales_yx=scales_yx,
    padding="max_detections",
    max_detections=frcnn_cfg.max_detections,
    return_tensors="pt",
)
features = output_dict.get("roi_features")
print(features)