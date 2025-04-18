# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.

import io

import numpy as np
import PIL.Image
import torch
from IPython.display import Image, display

from transformers import AutoTokenizer, VisualBertModel

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
model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Is there a cato?", return_tensors="pt")
# this is a custom function that returns the visual embeddings given the image path
visual_embeds = features 

visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
inputs.update(
    {
           "visual_embeds": visual_embeds,
           "visual_token_type_ids": visual_token_type_ids,
           "visual_attention_mask": visual_attention_mask,    }
)
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)  # (batch_size, sequence_length, hidden_size)
print(last_hidden_state) 