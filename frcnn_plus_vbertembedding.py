# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.

import io

import numpy as np
import PIL.Image
import torch
from IPython.display import Image, display
from umap import UMAP
from transformers import AutoTokenizer, VisualBertModel

from importlib.machinery import SourceFileLoader
utils = SourceFileLoader("utils", "NOTCODEDBYME/utils.py").load_module()
visual_cues = SourceFileLoader("visual_cues", "NOTCODEDBYME/visual_cues.py").load_module()
vis = SourceFileLoader("visa", "NOTCODEDBYME/visa.py").load_module()


class ProjectionLayer:
    def __init__(self):
        self.umap = UMAP(n_components=30)
    def forward(self, x):
        x = x.cpu().detach().numpy()
        x = self.umap.fit_transform(x)
        return torch.tensor(x).to(torch.float32).cuda(device=0)
    

class FRCNN_VisualBert_Embedding:
    def __init__(self):
        conf = utils.Config
        self.frcnn_cfg = conf.from_pretrained("unc-nlp/frcnn-vg-finetuned")

        self.frcnn = visual_cues.GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)
        self.ProjectionLayer = ProjectionLayer()
        self.image_preprocess = utils.Preprocess(self.frcnn_cfg)
    def forward(self, path, question):
        images, sizes, scales_yx = self.image_preprocess(path)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        features = output_dict.get("roi_features")
        model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        inputs = tokenizer(question, return_tensors="pt")
        
        visual_embeds = features
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float) 
        inputs.update(
        {
           "visual_embeds": visual_embeds,
           "visual_token_type_ids": visual_token_type_ids,
           "visual_attention_mask": visual_attention_mask,    }
    )

        Embedding = model(**inputs).last_hidden_state
        Embedding = Embedding[0, :, :]
        result = self.ProyectionLayer(Embedding)
        return result
    #
    def ProyectionLayer(self, Embedding):
        projection = self.ProjectionLayer.forward(Embedding)
        return projection
    
    def BuildImage(self, output):
        a = np.uint8(np.clip(output, 0, 255))
        f = io.BytesIO()
        PIL.Image.fromarray(a).save(f, "jpeg")
        display(Image(data=f.getvalue()))

    def draw_boxes(self, path):
        obj = "backbonelabelmap/objects_vocab.txt"
        attr = "backbonelabelmap/attributes_vocab.txt"
        objids = utils.get_data(obj)
        attrids = utils.get_data(attr)
        frcnn_visualizer = vis.SingleImageViz(path, id2obj=objids, id2attr=attrids)

        images, sizes, scales_yx = self.image_preprocess(path)
        output_dict = self.frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=self.frcnn_cfg.max_detections,
        return_tensors="pt",
        )


        frcnn_visualizer.draw_boxes(
        output_dict.get("boxes"),
        output_dict.pop("obj_ids"),
        output_dict.pop("obj_probs"),
        output_dict.pop("attr_ids"),
        output_dict.pop("attr_probs"),
        )
        return frcnn_visualizer._get_buffer()
    

    def Draw(self, path):
        try:
            a = self.draw_boxes(path)
            self.BuildImage(a)
        except Exception as e:
            print(e)
            print("Error in drawing boxes")






