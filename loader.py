import json
import mediapy as media
def LoadActivityNetDataset():
    with open("data/activitynet/0_30_s_activitynetqa_oe_qa_processed.json", "r") as f:
        data = json.load(f)
    return data
def LoadYoutubetDataset():
    with open("data/youtube/0_30_s_youtube_v0_1_cap_processed.json", "r") as f:
        data = json.load(f)
    return data
def LoadNextQA():
    with open("data/NextQA/0_30_s_nextqa_oe_qa_processed.json", "r") as f:
        data = json.load(f)
    return data

def ShowVideo(video_path, dataset):
    if dataset == "activitynet":
        video = media.read_video("data/activitynet/" + (video_path))
        media.show_video(video, title = "a", fps=5)
    elif dataset == "youtube":
        video = media.read_video("data/youtube/" + (video_path))
        media.show_video(video, title = "a", fps=5)
    elif dataset == "NextQA":
        video = media.read_video("data/NextQA/" + (video_path))
        media.show_video(video, title = "a", fps=5)
    else:
        raise ValueError("Invalid dataset name")
