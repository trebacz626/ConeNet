import torch

#create pl dataset that has image as input and 2 float values as target
class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder,  data, labels, transformations=None):
        self.root_folder = root_folder
        self.data = data
        self.labels = labels
        self.transformations = transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load image with PIL formate image_id to dddd
        image, label = self.data[idx], self.labels[idx]
        #to PIL
        # apply transformations
        if self.transformations:
            image = self.transformations(image)
        # return image and target
        return image, torch.tensor(label, dtype=torch.float32)

