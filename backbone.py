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
test_question = ["What can you see in the image?"]
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")

inputs = tokenizer(
    test_question,
    padding="max_length",
    max_length=20,
    truncation=True,
    return_token_type_ids=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt",
)



text = "What can you see in the image?"
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = features
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(features.shape[:-1])

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

labels = torch.tensor(1).unsqueeze(0)  # Batch size 1, Num choices 2

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits

print(outputs)