from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torch
from torchvision.transforms import transforms
class VOCObjectDetectionDataset(Dataset):
    def __init__(self, split: str = 'train', label2idx: dict = None, data_dir: str = None, list_transforms: list = [])-> Dataset[torch.tensor, torch.tensor, torch.tensor]:
        self.list_transforms = [transforms.ToTensor()]
        if list_transforms:
            self.list_transforms.extend(list_transforms)

        self.dataset = VOCDetection(
            root=data_dir,
            year='2007',
            image_set=split,
            download=True,
            transform=transforms.Compose(self.list_transforms),
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



# padding các ảnh trong batch theo kích thước của ảnh lớn nhất trong Batch, padding các GTboxes và GTlabels theo số lượng boxes lớn nhất trong Batch
def collate_fn(batch):
    imgs, targets  = zip(*batch)
    Hmax = max(img.shape[1] for img in imgs)
    Wmax = max(img.shape[2] for img in imgs)
    num_boxes_max = max(len(target['boxes']) for target in targets)
    GT_boxes = torch.zeros(len(imgs), num_boxes_max, 4)
    GT_labels = torch.zeros(len(imgs), num_boxes_max)
    mask_GT_boxes = torch.zeros(len(imgs), num_boxes_max, dtype = torch.bool)
    
    for i, target in enumerate(targets):
        num_boxes = len(target['boxes'])
        GT_boxes[i, :num_boxes, :] = target['boxes']
        GT_labels[i, :num_boxes] = target['labels']
        mask_GT_boxes[i,:num_boxes] = True

    img_padded = torch.zeros(len(imgs), 3, Hmax, Wmax)
    size_GT = [] # kích thước của ảnh thật
    for k, img in enumerate(imgs):
        img_padded[k,:,:img.shape[1],:img.shape[2]] = img
        size_GT.append([img.shape[1],img.shape[2]])
    return img_padded, GT_boxes, GT_labels, size_GT, mask_GT_boxes

