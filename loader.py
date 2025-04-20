import json
import mediapy as media
from tqdm import tqdm
import os
def LoadActivityNetDataset():
    with open("data/activitynet/0_30_s_activitynetqa_oe_qa_processed.json", "r") as f:
        data = json.load(f)
    return data
def LoadYoutubetDataset():
    with open("data/youtube/0_30_s_youtube_v0_1_cap_processed.json", "r") as f:
        data = json.load(f)
    return data
def LoadNextQA_OE():
    with open("data/NextQA/0_30_s_nextqa_oe_qa_processed.json", "r") as f:
        data = json.load(f)
    return data
def LoadNextQA_MC():
    with open("data/NextQA/0_30_s_nextqa_mc_qa_processed.json", "r") as f:
        data = json.load(f)
    return data
def LoadPerceptionDataset():
    with open("data/perception/0_30_s_perceptiontest_mc_qa_processed.json", "r") as f:
        data = json.load(f)
    return data

def LoadDatasets(rawdataYoutube, rawdataActivityNet, rawdataQA, rawdataQA2, rawdataperception):
    training_data = []
    labels_data = []
    names = []
    sources = []
    for i in tqdm(range(0, len(rawdataYoutube)), desc="Loading Youtube Dataset"):
        train = []
        t,l = rawdataYoutube[i]["conversations"][0]["value"].replace("<image>",""), rawdataYoutube[i]["conversations"][1]["value"].replace("<image>","")
        if len(t) >512 or len(l) >512:
            continue
        train.append(t)
        train.append("data/youtube/"+rawdataYoutube[i]["video"])
        training_data.append(train)
        labels_data.append(
        l
        )
        names.append(rawdataYoutube[i]["id"])
        sources.append("youtube")
    for i in tqdm(range(0, len(rawdataActivityNet)), desc="Loading ActivityNet open questions Dataset"):
        train = []
        t,l = rawdataActivityNet[i]["conversations"][0]["value"].replace("<image>",""), rawdataActivityNet[i]["conversations"][1]["value"].replace("<image>","")
        if len(t) > 512 or len(l) >512:
            continue
        train.append(t)
        train.append("data/activitynet/"+rawdataActivityNet[i]["video"])
        training_data.append(train)
        labels_data.append(
        l
        )
        names.append(rawdataActivityNet[i]["id"])
        sources.append("activitynet")
    for i in tqdm(range(0, len(rawdataQA)), desc="Loading Next QO-A Dataset"):
        train = []
        t,l = rawdataQA[i]["conversations"][0]["value"].replace("<image>",""), rawdataQA[i]["conversations"][1]["value"].replace("<image>","")
        if len(t) > 512 or len(l) >512:
            continue
        train.append(t)
        train.append("data/NextQA/"+rawdataQA[i]["video"])
        training_data.append(train)
        labels_data.append(
        l
        )
        names.append(rawdataQA[i]["id"].split("-")[1])
        sources.append("NEXTQA")
    
    for i in tqdm(range(0, len(rawdataQA2)), desc="Loading Next Q-A Dataset"):
        train = []
        t,l = rawdataQA2[i]["conversations"][0]["value"].replace("<image>",""), rawdataQA2[i]["conversations"][1]["value"].replace("<image>","")
        if len(t) > 512 or len(l) >512:
            continue
        train.append(t)
        train.append("data/NextQA/"+rawdataQA2[i]["video"])
        training_data.append(train)
        labels_data.append(
        l
        )
        names.append(rawdataQA2[i]["id"].split("-")[1])
        sources.append("NEXTQA")
    for i in tqdm(range(0, len(rawdataperception)), desc="Loading Perception Dataset"):
        train = []
        t,l = rawdataperception[i]["conversations"][0]["value"].replace("<image>",""), rawdataperception[i]["conversations"][1]["value"].replace("<image>","")
        if len(t) > 512 or len(l) >512:
            continue
        train.append(t)
        train.append("data/perception/"+rawdataperception[i]["video"])
        training_data.append(train)
        labels_data.append(
        l
        )
        names.append(rawdataperception[i]["id"].replace("perceptiontest_",""))
        sources.append("perception")
        return training_data, labels_data, names, sources
    

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
def STARTFROMI(i, training_data, model, buffer_dataset, sources, names, video_to_frames):
    try:
     for i in tqdm(range(0, len(training_data)), desc="Running Backbone on the training dataset and storing the results"):
      try:

         prompt = training_data[i][0]
         video = training_data[i][1]
         video_to_frames((video), "temp")
         if sources[i] == "youtube":

               for frame in os.listdir("temp/ytb_"+names[i]+".mp4"):
                  path = os.path.join("temp/ytb_"+names[i]+".mp4", frame)
                  Embedding = model.forward(path, prompt)
                  buffer_dataset.append(Embedding)
                  os.remove(path)
        
                  print("remaning frames "+str(len(os.listdir("temp/ytb_"+names[i]+".mp4"))))
         elif sources[i] == "activitynet":
        
               for frame in os.listdir("temp/"+names[i]+".mp4"):
                  path = os.path.join("temp/"+names[i]+".mp4", frame)
                  Embedding = model.forward(path, prompt)
                  buffer_dataset.append(Embedding)
                  os.remove(path)
           
                  print("remaning frames "+str(len(os.listdir("temp/"+names[i]+".mp4"))))
         elif sources[i] == "NEXTQA":
         
            for frame in os.listdir("temp/"+names[i]+".mp4"):
               path = os.path.join("temp/"+names[i]+".mp4", frame)
               Embedding = model.forward(path, prompt)
               buffer_dataset.append(Embedding)
               os.remove(path)
           
               print("remaning frames "+str(len(os.listdir("temp/"+names[i]+".mp4"))))
        
         elif sources[i] == "perception":
        
            for frame in os.listdir("temp/"+names[i]+".mp4"):
               path = os.path.join("temp/"+names[i]+".mp4", frame)
               Embedding = model.forward(path, prompt)
               buffer_dataset.append(Embedding)
               os.remove(path)
          
               print("remaning frames "+str(len(os.listdir("temp/"+names[i]+".mp4"))))
      except Exception as e:
         print(e)
         continue
    finally:
          return i, buffer_dataset
      