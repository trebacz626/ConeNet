import os

import pandas as pd
from tqdm import tqdm
import shutil

def main(split, root_folder, data_folder):
    df = pd.read_csv(f"{root_folder}/{split}.csv", header=1)

    # create folder root_folder/split/images
    if os.path.exists(f"{root_folder}/{split}/images"):
        shutil.rmtree(f"{root_folder}/{split}/images")
    if os.path.exists(f"{root_folder}/{split}/labels"):
        shutil.rmtree(f"{root_folder}/{split}/labels")

    os.makedirs(f"{root_folder}/{split}/images", exist_ok=True)
    os.makedirs(f"{root_folder}/{split}/labels", exist_ok=True)

    for row in tqdm(df.iterrows()):
        img_filename = f"{data_folder}/{row[1][0]}"
        text_filename = f"{data_folder}/{row[1][0].replace('jpg', 'txt')}"
        if os.path.exists(img_filename):
            shutil.copy(img_filename, f"{root_folder}/{split}/images/")
        if os.path.exists(text_filename):
            shutil.move(text_filename, f"{root_folder}/{split}/labels/")


if __name__ == "__main__":
    main("train", "./data/dataset_YOLO", "data/dataset_YOLO/YOLO_dataset")
    main("val", "./data/dataset_YOLO", "data/dataset_YOLO/YOLO_dataset")


