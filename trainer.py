from torch.utils.data import DataLoader
import os
from PIL import Image
import torch
from tqdm import tqdm_notebook as tqdm
from loader import *
from frcnn_plus_vbertembedding import FRCNN_VisualBert_Embedding
from videohandler import *
from IPython.display import clear_output

model = FRCNN_VisualBert_Embedding()
import shelve

rawdataYoutube = LoadYoutubetDataset()
rawdataActivityNet = LoadActivityNetDataset()
rawdataQA = LoadNextQA_OE()
rawdataQA2 = LoadNextQA_MC()
rawdataperception = LoadPerceptionDataset()
print("retrieved datasets, loading...")
training_data, labels_data, names, sources = LoadDatasets(rawdataYoutube, rawdataActivityNet, rawdataQA, rawdataQA2, rawdataperception)
print("Final dataset size of "+str(len(training_data))+" videos")
import warnings
warnings.filterwarnings("ignore")
   
if os.path.exists("buffer_dataset.pt"):
   buffer_dataset = torch.load("buffer_dataset.pt")
else:
   buffer_dataset = []
   buffer_dataset.append(0)
   torch.save(buffer_dataset,"buffer_dataset.pt")
current = buffer_dataset[0]
try :
   for i in tqdm(range(current, len(training_data)), desc="Running Backbone on the training dataset and storing the results"):
      current = i
      try:
         buffer_dataset[0] = current
         torch.save(buffer_dataset, "buffer_dataset.pt")
         prompt = training_data[i][0]
         video = training_data[i][1]
         video_to_frames((video), "temp")
         if sources[i] == "youtube":

               for frame in os.listdir("temp/ytb_"+names[i]+".mp4"):
                  path = os.path.join("temp/ytb_"+names[i]+".mp4", frame)
                  Embedding = model.forward(path, prompt)
                  buffer_dataset.append({Embedding, labels_data[i]})
                  os.remove(path)
        
                  print("remaning frames "+str(len(os.listdir("temp/ytb_"+names[i]+".mp4"))))
         elif sources[i] == "activitynet":
        
               for frame in os.listdir("temp/"+names[i]+".mp4"):
                  path = os.path.join("temp/"+names[i]+".mp4", frame)
                  Embedding = model.forward(path, prompt)
                  buffer_dataset.append({Embedding, labels_data[i]})
                  os.remove(path)
           
                  print("remaning frames "+str(len(os.listdir("temp/"+names[i]+".mp4"))))
         elif sources[i] == "NEXTQA":
         
            for frame in os.listdir("temp/"+names[i]+".mp4"):
               path = os.path.join("temp/"+names[i]+".mp4", frame)
               Embedding = model.forward(path, prompt)
               buffer_dataset.append({Embedding, labels_data[i]})
               os.remove(path)
           
               print("remaning frames "+str(len(os.listdir("temp/"+names[i]+".mp4"))))
        
         elif sources[i] == "perception":
        
            for frame in os.listdir("temp/"+names[i]+".mp4"):
               path = os.path.join("temp/"+names[i]+".mp4", frame)
               Embedding = model.forward(path, prompt)
               buffer_dataset.append({Embedding, labels_data[i]})
               os.remove(path)
          
               print("remaning frames "+str(len(os.listdir("temp/"+names[i]+".mp4"))))
      except Exception as e:
         print(e)
         continue
except Exception as e:
   print(e)
   pass
try:
   torch.save(buffer_dataset, "bufferdataset")
except Exception as e:
   pass
try:
   torch.save(buffer_dataset, "buffer_dataset.pt")
except Exception as e:
   pass
try:
   
   torch.save(buffer_dataset, "bufferdataset.pth")
except Exception as e:
   pass