import numpy as np
import cv2
import os
import click
import pandas as pd
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
from src.model.ColorModule import ColorModule
from src.model.simple_cnn import SimpleCNN

dataset_path = "./data/dataset_YOLO/YOLO_dataset"
model_path = "./checkpoints/SimpleCNN-epoch=09-validation_loss=0.05.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
ID_TO_COLOR = {
    0: (0, 165, 255), #orange
    1: (0, 255, 255), #yellow
    2: (255, 0, 0), #blue
}

transformations = torchvision.transforms.Compose(
        [ToTensor(), Resize((64, 64), antialias=True)])

checkpoint = torch.load(model_path)
print("Keys in the saved model:")
print(checkpoint['state_dict'].keys())

#read model from file
model = ColorModule(SimpleCNN(num_classes=3), num_classes=3)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

#iterate through all jpg files in folder
for filename in tqdm(os.listdir(dataset_path)):
    if filename.endswith(".txt"):
        #load image
        img = cv2.imread(os.path.join(dataset_path, filename.replace('.txt', '.jpg')))
        #bgr to rgb
        img_height, img_width = img.shape[:2]
        new_file_lines = []
        #iterate over lines in txt file
        with open(os.path.join(dataset_path, filename), 'r') as txt_file:
            for line in txt_file.readlines():
                #split line into list of strings
                line = line.split(' ')
                #convert list of strings to list of floats
                line = list(map(float, line))
                #convert normalized coordinates to pixel coordinates
                x = int(line[1] * img_width)
                y = int(line[2] * img_height)
                w = int(line[3] * img_width)
                h = int(line[4] * img_height)
                # #cut out image inside bounding box
                img_cut = img[y-h//2:y+h//2, x-w//2:x+w//2]
                img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)
                #apply transformations
                img_cut = transformations(img_cut)
                #predict color
                color = model(img_cut.unsqueeze(0).to(device)).to("cpu").detach()
                #argmax
                color = int(torch.argmax(color))
                #add bounding box to image with color
                cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), ID_TO_COLOR[color], 2)
                #add color to line
                line[0] = color
                #convert line back to list of strings
                line = list(map(str, line))
                #join list of strings to one string
                line = ' '.join(line)
                #add line to new file lines
                new_file_lines.append(line)
        #write new file lines to txt file
        with open(os.path.join(dataset_path, filename), 'w') as txt_file:
            txt_file.write('\n'.join(new_file_lines))








