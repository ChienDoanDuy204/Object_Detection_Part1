import torch
import torch.nn as nn
import timm
from torchvision.ops import box_iou

class OverFeatDetector(nn.Module):
    def __init__(self, num_classes:int = 1000)->None:
        super().__init__()
        self.backbone = timm.create_model(model_name='resnet18.a1_in1k', pretrained=True)
        self.backbone.global_pool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.ConvLayer = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=7, stride = 1),
            nn.ReLU(inplace=True)
        )
        self.act_fn = nn.ReLU(inplace=True)
        self.classifier_head = nn.Conv2d(in_channels=512, out_channels= num_classes, kernel_size=1, stride = 1)
        self.Regression_bbx_head = nn.Conv2d(in_channels=512, out_channels=4, kernel_size=1, stride =1)
    
    def forward(self,x:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.forward_features(x)
        x = self.ConvLayer(x)

        box_deltas = self.Regression_bbx_head(x) # dự đoán độ lệch box giữa anchor và GT_boxes
        logits = self.classifier_head(x) # Phân loại các ROI - Patch ảnh
        return logits, box_deltas
    


    # Tọa giả định vị trí ban đầu của bouding box chính là vị trí các ROI ảnh
    def make_anchors(self, O_h:int, O_w:int, stride = 32, window_size = 224):
        '''
        O_h: chiều cao của bản đồ feature map
        O_w: chiều rộng của bản đồ feature map
        stride: stride của backbone
        window_size: kích thước của ảnh input
        '''
        ys = torch.arange(O_h)*stride # Các tạo độ y1 của từng ROI [0, 32, 64, ...]
        xs = torch.arange(O_w)*stride # Các toạ độ x1 của từng ROI [0, 32, 64, ...]
        # tạo ra tọa độ tất các các góc trái trên của các ROI 
        # y1 = [[0,  0,  0],
        #       [32, 32, 32]
        #       [64, 64, 64]]
        # x1 = [[0, 32, 64],
        #       [0, 32, 64]
        #       [0, 32, 64]]
        y1, x1 = torch.meshgrid(ys, xs, indexing='ij')
        anchors = torch.stack([x1, y1, x1+window_size, y1+window_size], dim=-1) # shape (O_h, O_w, 4 - tọa độ xmin, ymin, xmax, ymax)
        anchors = anchors.view(-1, 4) # Flatten the anchors to a 2D tensor of shape (M, 4)
        return anchors
    
    # tính độ lệch box_delta trên các vùng dương giữa anchor và GT_box
    def encode_delta(self, anchor: torch.tensor, gt_box: torch.tensor)-> torch.tensor:
        w_anchor, h_anchor = anchor[:,2]-anchor[:,0], anchor[:,3]-anchor[:,1]
        xcenter_anchor, ycenter_anchor = anchor[:,0] + 0.5*w_anchor, anchor[:,1] + 0.5*h_anchor

        w_gt, h_gt = gt_box[:,2]-gt_box[:,0], gt_box[:,3]-gt_box[:,1]
        xcenter_gt, ycenter_gt = gt_box[:,0] + 0.5*w_gt, gt_box[:,1] + 0.5*h_gt

        return torch.tensor([(xcenter_gt-xcenter_anchor)/w_anchor, (ycenter_gt-ycenter_anchor)/h_anchor, torch.log(w_gt/w_anchor), torch.log(h_gt/h_anchor)]).T
    
    def build_target(self, anchors: torch.tensor, GT_boxes: torch.tensor, GT_labels: torch.tensor, positive_threshold: float = 0.6, negative_threshold: float = 0.3):
        '''
        anchors: bbx mà mô hình sự đoán cho từng ROI - Patch ảnh (M, 4)
        GT_boxes: Tọa độ các bbx ground truth của ảnh(N, 4)
        GT_labels: Nhãn của các GT_boxes tương ứng (N)
        positive_threshold: Ngưỡng IoU để xác định ROI nào là positive
        negative_threshold: Ngưỡng IoU để xác định ROI nào là negative - background
        '''
        M = anchors.shape[0] # Tổng số anchors của mỗi ảnh
        labels = torch.zeros(M, dtype = torch.int64) # mặc định gán class của các anchors là nền - background
        target_delta_bbx = torch.zeros(M,4)

        # Kiểm tra xem ảnh có GT bounding box không
        if GT_boxes.numel() == 0:
            return labels, target_delta_bbx # trả về nhãn thực tế là back gound và độ lệch thực tế giữa anchor và GT_boxes bằng 0
        

        iou_score_matrix = box_iou(anchors, GT_boxes) #(M,N) tính IoU của mỗi bbx anchor với tất cả các bbx GT
        max_iou_score, idx_max = torch.max(iou_score_matrix, dim = 1) # tìm M giá trị IoU lớn nhất và index của bbx GT tương ứng

        positive_mask = max_iou_score>= positive_threshold # Tìm các anchors có IoU >= positive_threshold - mask positive

        labels[positive_mask] = GT_labels[idx_max[positive_mask]] # Gán nhãn của GT bbx cho anchor thỏa mãn ngưỡng overlap > pos_threshold

        labels[(max_iou_score>negative_threshold) & (max_iou_score<positive_threshold)] = -1 

        # Tính độ lệch của bbx positive so với GT bbx
        target_delta_bbx[positive_mask] = self.encode_delta(anchors[positive_mask], GT_boxes[idx_max[positive_mask]])