import torch
from torch.utils.data import Dataset
import cv2
import os
from torch.nn.utils.rnn import pad_sequence

class DatasetFromImages(Dataset):
    # Loader from train_list.csv
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.file_names = df['img_list'].values
        self.labels = df['label'].values
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def read_images(self, img_list, transform=None):
        X = []
        img_list = img_list.split(',')
        for file_name in img_list:
            file_path = os.path.join(self.data_path, file_name)
            image = cv2.imread(file_path)

            if transform:
                augmented = self.transform(image=image)
                image = augmented['image']

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, idx):
        img_list = self.file_names[idx]
        X = self.read_images(img_list, self.transform)
        y = torch.tensor(self.labels[idx]).long()
        return X, y


class KeyFrameDataset(Dataset):
    # Loader from train_list_keyframes.csv
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.video_names = df['id'].values
        self.labels = df['label'].values
        self.keyframes = df['keyframes'].values
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def read_images(self, video_name, keyframes, transform=None):
        X = []
        path_to_keyframes = video_name.replace('_', '/')
        keyframes = keyframes[1:-1].split(',')
        for file_name in keyframes:
            if int(file_name)<10:
                file_path = os.path.join(self.data_path, path_to_keyframes, 'frame_00'+str(int(file_name))+'.jpg')
            elif int(file_name)<100:
                file_path = os.path.join(self.data_path, path_to_keyframes, 'frame_0'+str(int(file_name))+'.jpg')
            else:
                file_path = os.path.join(self.data_path, path_to_keyframes, 'frame_' +str(int(file_name))+ '.jpg')
            image = cv2.imread(file_path)

            if transform:
                augmented = self.transform(image=image)
                image = augmented['image']

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        keyframes = self.keyframes[idx]
        X = self.read_images(video_name, keyframes, self.transform)
        y = torch.tensor(self.labels[idx]).long()
        return X, y

class SingleFrameDataset(Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.folders = df['id'].values
        self.file_names = df['frame'].values
        self.labels = df['label'].values
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        folder = self.folders[idx].replace('_', '/')
        file_name = self.file_names[idx]
        if int(file_name) < 10:
            file_path = os.path.join(self.data_path, folder, 'frame_00' + str(file_name) + '.jpg')
        elif int(file_name) < 100:
            file_path = os.path.join(self.data_path, folder, 'frame_0' + str(file_name) + '.jpg')
        else:
            file_path = os.path.join(self.data_path, folder, 'frame_' + str(file_name) + '.jpg')
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  # x_lens = [len(x) for x in xx]
  # y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  # yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

  return xx_pad, yy
