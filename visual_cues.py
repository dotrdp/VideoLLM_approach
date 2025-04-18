from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

img = decode_image("TEST.jpg")

# Step 1: Initialize model with the best available weights


# Step 2: Initialize the inference transforms


# Step 3: Apply inference preprocessing transforms


# Step 4: Use the model and visualize the prediction


class FRCNN:
    def __init__(self):
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, threshold=0.9)
        print(self.model.roi_heads)
        self.model.eval()
        self.preprocess = self.weights.transforms()

    def drawboxes(self, img):
        img = decode_image(img)
        batch = [self.preprocess(img)]
        prediction = self.model(batch)[0]
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                                  labels=labels,
                                  colors="red",
                                  width=4, font_size=30)
        return to_pil_image(box.detach())
    def get_visual_embeddings(self, img):
        img = decode_image(img)
        batch = [self.preprocess(img)]
        prediction = self.model(batch)[0]
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                                  labels=labels,
                                  colors="red",
                                  width=4, font_size=30)
        return box