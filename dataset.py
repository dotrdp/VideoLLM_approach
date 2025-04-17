from datasets import load_dataset
from torch.utils.data import DataLoader

def LOADDATASET():
    # Load the dataset from Hugging Face
    # The dataset is streamed to save memory
    # and the DataLoader is set up for efficient data loading
    # with multiple workers and prefetching.
    
    # Load the dataset using streaming mode
    ds = load_dataset('https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K', streaming=True)
    dl = DataLoader(ds, num_workers=1, prefetch_factor=2, batch_size=1, pin_memory=True)
    return dl