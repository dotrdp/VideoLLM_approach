# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
import torch
from visual_cues import *

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForQuestionAnswering.from_pretrained("visualbert-vqa")

text = "Is there a cat in the image?"

inputs = tokenizer(
    text,
    padding="max_length",
    max_length=20,
    truncation=True,
    return_token_type_ids=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt",
)
model2 = FRCNN()
visual_embeds = model2.get_visual_embeddings("TEST.jpg")
output_vqa = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_embeds=features,
        visual_attention_mask=torch.ones(features.shape[:-1]),
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    # get prediction
pred_vqa = output_vqa["logits"].argmax(-1)
print("Question:", text)
print("prediction from VisualBert VQA:", pred_vqa)