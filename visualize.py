from src.data.ColorDataset import ColorDataset
import numpy as np
import cv2

images = np.load('./data/dataset_color/data.npy')
labels = np.load('./data/dataset_color/labels.npy')

for image, label in zip(images, labels):
    print(label)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("image" , image)
    cv2.waitKey(0)


#orange
#yellow
#blue
