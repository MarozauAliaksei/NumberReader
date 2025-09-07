import os
import cv2
import torch
from torch.utils.data import Dataset
from config import char2idx, IMG_HEIGHT, IMG_WIDTH

class OCRDataset(Dataset):
    def __init__(self, root):
        self.root = root
        files = sorted([f for f in os.listdir(root) if f.lower().endswith((".jpg",".png",".jpeg"))])
        self.files = []
        self.labels = []
        for f in files:
            label = os.path.splitext(f)[0].split("_")[0]
            if all(c in char2idx for c in label):
                self.files.append(f)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root,self.files[idx])
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        h,w = img.shape
        new_w = int(w*IMG_HEIGHT/h)
        img = cv2.resize(img,(new_w,IMG_HEIGHT))
        _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = torch.tensor(img,dtype=torch.float32).unsqueeze(0)/255.0
        if img.shape[2]<IMG_WIDTH:
            pad = IMG_WIDTH-img.shape[2]
            img = torch.nn.functional.pad(img,(0,pad,0,0))
        elif img.shape[2]>IMG_WIDTH:
            img = img[:,:,:IMG_WIDTH]
        label = torch.tensor([char2idx[c] for c in self.labels[idx]],dtype=torch.long)
        return img,label,len(label)

def collate_fn(batch):
    imgs, labels, label_lengths = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths,dtype=torch.long)
    return imgs,labels,label_lengths
