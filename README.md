## Decoder + VisualBert chain of continous thought approach
For this, I approached video by proccessing it frame by frame, generating a context-aware embedding along with roi-features using a FPN+CNN backbone, furthermore It was plugged in to a pretrained Visual-Bert model on VQA. Visual Bert chain of continous thought embeddings were used in a decoder to get an answer.
<img width="669" alt="Screenshot 2025-05-17 at 2 21 18 a m" src="https://github.com/user-attachments/assets/b510da24-4801-4648-811a-cda57c96dab5" />
