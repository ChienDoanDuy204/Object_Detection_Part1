import torch
import torch.nn as nn
import timm
from torchvision.ops import box_iou
from tqdm import tqdm
from torch.amp import GradScaler

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
    def make_anchors(self, O_h:int, O_w:int, batch_size: int, stride = 32, window_size = 224):
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
        
        anchors = anchors.unsqueeze(0).repeat(batch_size,1,1) # shape(B, M, 4)
        return anchors
    
    # tính độ lệch box_delta trên các vùng dương giữa anchor và GT_box
    def encode_delta(self, anchor: torch.tensor, gt_box: torch.tensor)-> torch.tensor:
        w_anchor, h_anchor = anchor[:,:,2]-anchor[:,:,0], anchor[:,:,3]-anchor[:,:,1] # w_anchor shape(B, M)
        xcenter_anchor, ycenter_anchor = anchor[:,:,0] + 0.5*w_anchor, anchor[:,:,1] + 0.5*h_anchor

        w_gt, h_gt = gt_box[:,:,2]-gt_box[:,:,0], gt_box[:,:,3]-gt_box[:,:,1]
        xcenter_gt, ycenter_gt = gt_box[:,:,0] + 0.5*w_gt, gt_box[:,:,1] + 0.5*h_gt

        return torch.stack([(xcenter_gt-xcenter_anchor)/w_anchor, (ycenter_gt-ycenter_anchor)/h_anchor, torch.log(w_gt/w_anchor), torch.log(h_gt/h_anchor)], dim =2)
    
    # Giải mã độ lệch box_delta mà mô hình dự đoán để trả về tọa độ bounding box thực tế [xmin, ymin, xmax, ymax]
    def decode_delta(self, anchor: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Giải mã độ lệch (delta) mà mô hình dự đoán kết hợp với anchor boxes
        để trả về tọa độ bounding box thực tế [xmin, ymin, xmax, ymax].

        Args:
            anchor: Tensor tọa độ của anchor boxes [x1, y1, x2, y2], shape (B, M, 4) hoặc (..., 4)
            delta: Tensor độ lệch dự đoán [dx, dy, dw, dh], shape (B, M, 4) hoặc (..., 4)

        Returns:
            pred_boxes: Tensor tọa độ bounding box thực tế [xmin, ymin, xmax, ymax], shape (B, M, 4) hoặc (..., 4)
        """
        w_anchor = anchor[..., 2] - anchor[..., 0]
        h_anchor = anchor[..., 3] - anchor[..., 1]
        xcenter_anchor = anchor[..., 0] + 0.5 * w_anchor
        ycenter_anchor = anchor[..., 1] + 0.5 * h_anchor

        dx = delta[..., 0]
        dy = delta[..., 1]
        dw = torch.clamp(delta[..., 2], min=-10.0, max=10.0)
        dh = torch.clamp(delta[..., 3], min=-10.0, max=10.0)

        xcenter_pred = xcenter_anchor + dx * w_anchor
        ycenter_pred = ycenter_anchor + dy * h_anchor
        w_pred = w_anchor * torch.exp(dw)
        h_pred = h_anchor * torch.exp(dh)

        xmin = xcenter_pred - 0.5 * w_pred
        ymin = ycenter_pred - 0.5 * h_pred
        xmax = xcenter_pred + 0.5 * w_pred
        ymax = ycenter_pred + 0.5 * h_pred

        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    
    def build_target(self, anchors: torch.Tensor, GT_boxes: torch.Tensor, GT_labels: torch.Tensor, positive_threshold: float = 0.6, negative_threshold: float = 0.3, mask_GT_boxes: torch.Tensor = None):
        '''
        anchors: bbx mà mô hình dự đoán cho từng ROI - Patch ảnh (B, M, 4)
        GT_boxes: Tọa độ các bbx ground truth của ảnh (B, N, 4)
        GT_labels: Nhãn của các GT_boxes tương ứng (B, N)
        positive_threshold: Ngưỡng IoU để xác định ROI nào là positive
        negative_threshold: Ngưỡng IoU để xác định ROI nào là negative - background
        mask_GT_boxes: Mask để loại bỏ các GT_boxes đã được padding (B, N)
        '''
        B = anchors.shape[0]
        M = anchors.shape[1] # Tổng số anchors của mỗi ảnh
        device = anchors.device
        labels = torch.zeros((B, M), dtype=torch.int64, device=device) # mặc định gán class của các anchors là nền - background
        target_delta_bbx = torch.zeros((B, M, 4), dtype=torch.float32, device=device)

        for b in range(B):
            anchors_b = anchors[b]  # (M, 4)

            # Lọc các GT boxes hợp lệ của ảnh thứ b (loại bỏ padding)
            if mask_GT_boxes is not None:
                valid_mask = mask_GT_boxes[b].bool()
                gt_boxes_b = GT_boxes[b][valid_mask]
                gt_labels_b = GT_labels[b][valid_mask]
            else:
                gt_boxes_b = GT_boxes[b]
                gt_labels_b = GT_labels[b]

            if gt_boxes_b.shape[0] == 0:
                continue

            # Tính IoU giữa M anchors và K GT boxes thật: shape (M, K)
            iou_score_matrix = box_iou(anchors_b, gt_boxes_b)
            max_iou_score, idx_max = torch.max(iou_score_matrix, dim=1)  # (M,)

            positive_mask = max_iou_score >= positive_threshold
            ignore_mask = (max_iou_score > negative_threshold) & (max_iou_score < positive_threshold)

            # Gán nhãn ignore = -1
            labels[b, ignore_mask] = -1

            # Gán nhãn và tính delta cho positive anchors
            if positive_mask.any():
                labels[b, positive_mask] = gt_labels_b[idx_max[positive_mask]].long()

                pos_anchors = anchors_b[positive_mask].unsqueeze(0)  # shape (1, num_pos, 4)
                matched_gt = gt_boxes_b[idx_max[positive_mask]].unsqueeze(0)  # shape (1, num_pos, 4)
                target_delta_bbx[b, positive_mask] = self.encode_delta(pos_anchors, matched_gt).squeeze(0)

        return labels, target_delta_bbx


class TrainModel(nn.Module):
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, n_epochs: int = 30, learning_rate: float = 0.001, batch_size: int = 32)->None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.backbone.parameters(), 'lr': learning_rate/10, 'weight_decay': 0.0005},
            {'params': self.model.ConvLayer.parameters(), 'lr': learning_rate, 'weight_decay': 0.0005},
            {'params': self.model.classifier_head.parameters(), 'lr': learning_rate, 'weight_decay': 0.0005},
            {'params': self.model.Regression_bbx_head.parameters(), 'lr': learning_rate, 'weight_decay': 0.0005}
        ])
        self.classifier_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        self.regression_loss = nn.SmoothL1Loss(reduction='sum')
        self.n_epochs = n_epochs
        self.dataloader = dataloader
        self.batch_size = batch_size
    def get_accuracy(self, logits: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Tính độ chính xác phân loại của các anchors hợp lệ (loại trừ các vị trí ignore -1).
        """
        mask = (y_true != -1)
        if mask.sum() == 0:
            return 0.0
        y_pred = torch.argmax(logits, dim=-1)
        correct_tokens = (y_pred == y_true) & mask
        return (correct_tokens.sum() / mask.sum()).item()

    def forward(self):
        """
        Hàm huấn luyện mô hình Object Detection qua n_epochs với Mixed Precision (AMP),
        tính toán multi-task loss (Classification Loss + Bounding Box Regression Loss) và cập nhật trọng số.
        """
        self.TrainLosses = []
        self.TrainAcc = []
        scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in tqdm(range(self.n_epochs), desc="Training"):
            self.model.train()
            loss_epoch = 0.0
            acc_epoch = 0.0
            total_valid_tokens_epoch = 0

            for batch in self.dataloader:
                imgs = batch[0].to(self.device)
                gt_boxes = batch[1].to(self.device)
                gt_labels = batch[2].to(self.device)
                mask_gt_boxes = batch[4].to(self.device) if len(batch) > 4 else None

                with torch.autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu'):
                    # 1. Forward qua mô hình
                    logits, box_deltas = self.model(imgs)
                    B, num_classes, Oh, Ow = logits.shape
                    M = Oh * Ow

                    # Reshape logits -> (B, M, num_classes) và box_deltas -> (B, M, 4)
                    logits = logits.permute(0, 2, 3, 1).contiguous().view(B, M, num_classes)
                    box_deltas = box_deltas.permute(0, 2, 3, 1).contiguous().view(B, M, 4)

                    # 2. Sinh lưới anchors
                    anchors = self.model.make_anchors(
                        O_h=Oh,
                        O_w=Ow,
                        batch_size=B,
                        stride=32,
                        window_size=224
                    ).to(self.device)

                    # 3. Tạo target labels và target deltas cho các anchors bằng hàm build_target của OverFeatDetector
                    target_labels, target_delta_bbx = self.model.build_target(
                        anchors=anchors,
                        GT_boxes=gt_boxes,
                        GT_labels=gt_labels,
                        positive_threshold=0.6,
                        negative_threshold=0.3,
                        mask_GT_boxes=mask_gt_boxes
                    )

                    # 4. Tính Classification Loss (CrossEntropyLoss với ignore_index=-1)
                    cls_loss = self.classifier_loss(
                        logits.view(-1, num_classes),
                        target_labels.view(-1)
                    )

                    # 5. Tính Bounding Box Regression Loss trên các positive anchors
                    pos_mask = (target_labels > 0)
                    num_pos = pos_mask.sum().item()
                    if num_pos > 0:
                        reg_loss = self.regression_loss(
                            box_deltas[pos_mask],
                            target_delta_bbx[pos_mask]
                        )
                    else:
                        reg_loss = torch.tensor(0.0, device=self.device)

                    # 6. Chuẩn hóa loss (vì loss được cấu hình reduction='sum')
                    valid_mask = (target_labels != -1)
                    num_valid = valid_mask.sum().item()

                    loss = (cls_loss / max(1, num_valid)) + (reg_loss / max(1, num_pos))
                    acc = self.get_accuracy(logits, target_labels)

                # 7. Backward và cập nhật trọng số
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_epoch += loss.item() * num_valid
                acc_epoch += acc * num_valid
                total_valid_tokens_epoch += num_valid

            epoch_loss = loss_epoch / max(1, total_valid_tokens_epoch)
            epoch_acc = acc_epoch / max(1, total_valid_tokens_epoch)
            self.TrainLosses.append(epoch_loss)
            self.TrainAcc.append(epoch_acc)

            tqdm.write(f"Epoch [{epoch + 1}/{self.n_epochs}] - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc * 100:.2f}%")

