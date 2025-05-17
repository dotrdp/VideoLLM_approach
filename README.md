## Decoder + VisualBert chain of continous thought approach
For this, I approached video by proccessing it frame by frame, generating a context-aware embedding along with roi-features using a FPN+CNN backbone, furthermore It was plugged in to a pretrained Visual-Bert model on VQA. Visual Bert chain of continous thought embeddings were used in a decoder to get an answer.
