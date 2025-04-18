from datasets import load_dataset


def dat():
    # Load the dataset from Hugging Face
    # The dataset is streamed to save memory
    # and the DataLoader is set up for efficient data loading
    # with multiple workers and prefetching.
    
    # Load the dataset using streaming mode
    ds = load_dataset('lmms-lab/LLaVA-Video-178K','0_30_s_nextqa' , streaming=True)
    return ds