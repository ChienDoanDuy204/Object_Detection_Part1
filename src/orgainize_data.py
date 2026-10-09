from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torch
from torchvision.transforms import transforms
import math
class VOCObjectDetectionDataset(Dataset):
    def __init__(self, split: str = 'train', label2idx: dict = None, data_dir: str = None, list_transforms: list = [transforms.ToTensor()])-> Dataset[torch.tensor, torch.tensor, torch.tensor]:
        self.dataset = VOCDetection(
            root=data_dir,
            year='2007',
            image_set=split,
            download=True,
            transform=transforms.Compose(list_transforms),
        )
        self.label2idx = label2idx
    def __len__(self)-> int:
        return len(self.dataset)
    def __getitem__(self,index):
        img, tgt = self.dataset[index]
        boxes , labels = [],[]

        for obj in tgt['annotation']['object']:
            labels.append(self.label2idx[obj['name']])
            boxes.append([float(obj['bndbox']['xmin']),float(obj['bndbox']['ymin']),float(obj['bndbox']['xmax']),float(obj['bndbox']['ymax'])])
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)

        target = {"boxes": boxes, "labels": labels}
        return img, target



# padding các ảnh trong batch theo kích thước của ảnh lớn nhất trong Batch
def collate_fn(batch, divisible : int = 32):
    imgs, targets  = zip(*batch)
    Hmax = max(img.shape[1] for img in imgs)
    Wmax = max(img.shape[2] for img in imgs)

    H_pad = int(math.ceil(Hmax/ divisible))*divisible
    W_pad = int(math.ceil(Wmax/ divisible))*divisible
    img_padded = torch.zeros(len(imgs), 3, Hmax, Wmax)
    size_GT = [] # kích thước của ảnh thật
    for k, img in enumerate(imgs):
        img_padded[k,:,:img.shape[1],:img.shape[2]] = img
        size_GT.append([img.shape[1],img.shape[2]])
    return torch.tensor(img_padded), list(targets), size_GT

