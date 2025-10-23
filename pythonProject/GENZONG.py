import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from PIL import Image
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# SiamFC网络定义
class SiamFC(nn.Module):
    def __init__(self):
        super(SiamFC, self).__init__()
        # 特征提取网络
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256)
        )

    def forward(self, z, x):
        # z: 模板图像 (127x127)
        # x: 搜索区域 (255x255)
        z_feat = self.feature_extract(z)  # [batch, 256, 6, 6]
        x_feat = self.feature_extract(x)  # [batch, 256, 22, 22]

        # 互相关操作
        batch_size = z_feat.size(0)
        out_channel = z_feat.size(1)
        response = nn.functional.conv2d(
            x_feat.view(1, batch_size * out_channel, x_feat.size(2), x_feat.size(3)),
            z_feat.view(batch_size * out_channel, 1, z_feat.size(2), z_feat.size(3)),
            groups=batch_size
        )
        response = response.view(batch_size, 1, response.size(2), response.size(3))

        return response


# 数据集类
class OTB50Dataset(Dataset):
    def __init__(self, data_root, transform=None, train=True):
        self.data_root = data_root
        self.transform = transform
        self.train = train

        # 加载OTB50数据集
        self.sequences = self.load_sequences()
        self.samples = self.prepare_samples()

    def load_sequences(self):
        sequences = []
        seq_dirs = glob.glob(os.path.join(self.data_root, "*"))
        for seq_dir in seq_dirs:
            if os.path.isdir(seq_dir):
                img_files = sorted(glob.glob(os.path.join(seq_dir, "img", "*.jpg")))
                if len(img_files) > 0:
                    # 读取ground truth
                    gt_file = os.path.join(seq_dir, "groundtruth_rect.txt")
                    if os.path.exists(gt_file):
                        with open(gt_file, 'r') as f:
                            gt_lines = f.readlines()
                        gt_rects = []
                        for line in gt_lines:
                            if ',' in line:
                                coords = list(map(float, line.strip().split(',')))
                            else:
                                coords = list(map(float, line.strip().split()))
                            if len(coords) >= 4:
                                gt_rects.append(coords[:4])

                        if len(gt_rects) == len(img_files):
                            sequences.append({
                                'name': os.path.basename(seq_dir),
                                'img_files': img_files,
                                'gt_rects': gt_rects
                            })
        return sequences

    def prepare_samples(self):
        samples = []
        for seq in self.sequences:
            for i in range(len(seq['img_files']) - 1):
                samples.append({
                    'seq_name': seq['name'],
                    'template_idx': i,
                    'search_idx': i + 1,
                    'template_rect': seq['gt_rects'][i],
                    'search_rect': seq['gt_rects'][i + 1]
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def crop_and_resize(self, img, bbox, size):
        """根据边界框裁剪并调整图像大小"""
        x, y, w, h = bbox
        center_x, center_y = x + w / 2, y + h / 2

        # 扩展边界框
        context = 0.5 * (w + h)
        crop_size = 2 * context

        # 计算裁剪区域
        crop_x1 = max(0, center_x - crop_size / 2)
        crop_y1 = max(0, center_y - crop_size / 2)
        crop_x2 = min(img.width, center_x + crop_size / 2)
        crop_y2 = min(img.height, center_y + crop_size / 2)

        # 裁剪图像
        crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # 调整大小
        resized_img = crop_img.resize((size, size), Image.BILINEAR)

        return resized_img

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载模板图像 (第t帧)
        template_img = Image.open(sample['img_files'][sample['template_idx']])
        template_crop = self.crop_and_resize(template_img, sample['template_rect'], 127)

        # 加载搜索图像 (第t+1帧)
        search_img = Image.open(sample['img_files'][sample['search_idx']])
        search_crop = self.crop_and_resize(search_img, sample['search_rect'], 255)

        if self.transform:
            template_crop = self.transform(template_crop)
            search_crop = self.transform(search_crop)

        return template_crop, search_crop


# 训练函数
def train_siamfc(model, dataloader, criterion, optimizer, device, num_epochs=50):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (template, search) in enumerate(progress_bar):
            template = template.to(device)
            search = search.to(device)

            # 前向传播
            response = model(template, search)

            # 创建目标响应图 (高斯分布)
            batch_size, _, h, w = response.shape
            target = create_target_response(h, w, batch_size).to(device)

            # 计算损失
            loss = criterion(response, target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

    return model


def create_target_response(h, w, batch_size):
    """创建目标响应图（高斯分布）"""
    target = torch.zeros(batch_size, 1, h, w)

    # 在中心位置创建高斯分布
    center_x, center_y = w // 2, h // 2
    sigma = 0.1 * min(h, w)

    for i in range(h):
        for j in range(w):
            dist = ((i - center_y) ** 2 + (j - center_x) ** 2) / (2 * sigma ** 2)
            target[:, :, i, j] = torch.exp(-dist)

    return target


# 测试和跟踪函数
def track_sequence(model, sequence, device, output_video_path=None):
    """在单个序列上运行跟踪"""
    model.eval()

    img_files = sequence['img_files']
    gt_rects = sequence['gt_rects']

    # 初始化跟踪结果
    tracked_rects = [gt_rects[0]]  # 第一帧使用ground truth

    # 准备视频输出
    if output_video_path:
        first_img = cv2.imread(img_files[0])
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (w, h))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    template_img = Image.open(img_files[0])
    template_bbox = gt_rects[0]

    for i in tqdm(range(1, len(img_files)), desc=f"Tracking {sequence['name']}"):
        # 获取模板特征
        template_crop = crop_and_resize_pil(template_img, template_bbox, 127)
        template_tensor = transform(template_crop).unsqueeze(0).to(device)

        # 当前帧
        current_img = Image.open(img_files[i])

        # 基于上一帧位置生成搜索区域
        prev_bbox = tracked_rects[-1]
        search_crop = crop_and_resize_pil(current_img, prev_bbox, 255)
        search_tensor = transform(search_crop).unsqueeze(0).to(device)

        # 前向传播
        with torch.no_grad():
            response = model(template_tensor, search_tensor)

        # 找到响应最大的位置
        response_map = response.squeeze().cpu().numpy()
        h, w = response_map.shape

        # 找到最大响应位置
        max_pos = np.unravel_index(np.argmax(response_map), response_map.shape)

        # 将响应图位置转换回原图坐标
        new_bbox = convert_response_to_bbox(max_pos, prev_bbox, h, w, 255)
        tracked_rects.append(new_bbox)

        # 更新模板（简单策略：每10帧更新一次）
        if i % 10 == 0:
            template_img = current_img
            template_bbox = new_bbox

        # 可视化当前帧
        if output_video_path:
            img = cv2.imread(img_files[i])
            # 绘制跟踪结果
            x, y, w, h = map(int, new_bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制ground truth
            gt_x, gt_y, gt_w, gt_h = map(int, gt_rects[i])
            cv2.rectangle(img, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (255, 0, 0), 2)

            # 添加文本
            cv2.putText(img, 'Tracked', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, 'Ground Truth', (gt_x, gt_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            out.write(img)

    if output_video_path:
        out.release()

    return tracked_rects


def crop_and_resize_pil(img, bbox, size):
    """PIL版本的裁剪和调整大小"""
    x, y, w, h = bbox
    center_x, center_y = x + w / 2, y + h / 2

    # 扩展边界框
    context = 0.5 * (w + h)
    crop_size = 2 * context

    # 计算裁剪区域
    crop_x1 = max(0, center_x - crop_size / 2)
    crop_y1 = max(0, center_y - crop_size / 2)
    crop_x2 = min(img.width, center_x + crop_size / 2)
    crop_y2 = min(img.height, center_y + crop_size / 2)

    # 裁剪图像
    crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # 调整大小
    resized_img = crop_img.resize((size, size), Image.BILINEAR)

    return resized_img


def convert_response_to_bbox(response_pos, prev_bbox, response_h, response_w, search_size):
    """将响应图位置转换回原图边界框"""
    pos_y, pos_x = response_pos

    # 将响应图坐标映射到搜索区域坐标
    scale = search_size / response_w
    search_center_x = pos_x * scale
    search_center_y = pos_y * scale

    # 搜索区域在原图中的位置
    prev_x, prev_y, prev_w, prev_h = prev_bbox
    prev_center_x = prev_x + prev_w / 2
    prev_center_y = prev_y + prev_h / 2

    context = 0.5 * (prev_w + prev_h)
    crop_size = 2 * context

    # 计算新边界框中心
    new_center_x = prev_center_x + (search_center_x - search_size / 2) * (crop_size / search_size)
    new_center_y = prev_center_y + (search_center_y - search_size / 2) * (crop_size / search_size)

    # 保持相同的大小
    new_bbox = [new_center_x - prev_w / 2, new_center_y - prev_h / 2, prev_w, prev_h]

    return new_bbox


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据路径
    data_root = 'E:/OTB50'  # 请修改为您的OTB50数据集路径

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集和数据加载器
    dataset = OTB50Dataset(data_root, transform=transform, train=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    print(f'Dataset size: {len(dataset)}')
    print(f'Number of sequences: {len(dataset.sequences)}')

    # 创建模型
    model = SiamFC().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("Starting training...")
    model = train_siamfc(model, dataloader, criterion, optimizer, device, num_epochs=10)

    # 保存模型
    torch.save(model.state_dict(), 'siamfc_otb50.pth')
    print("Model saved to siamfc_otb50.pth")

    # 测试模型
    print("\nStarting testing...")

    # 加载测试序列
    test_sequences = dataset.sequences[:5]  # 使用前5个序列进行测试

    for seq in test_sequences:
        print(f"Testing on sequence: {seq['name']}")

        # 创建输出目录
        os.makedirs('output_videos', exist_ok=True)
        output_path = f"output_videos/{seq['name']}_tracking.avi"

        # 运行跟踪
        tracked_rects = track_sequence(model, seq, device, output_path)

        print(f"Tracking results saved to {output_path}")

        # 计算精度（简单的中心位置误差）
        gt_rects = seq['gt_rects']
        center_errors = []
        for i in range(len(gt_rects)):
            gt_x, gt_y, gt_w, gt_h = gt_rects[i]
            track_x, track_y, track_w, track_h = tracked_rects[i]

            gt_center = (gt_x + gt_w / 2, gt_y + gt_h / 2)
            track_center = (track_x + track_w / 2, track_y + track_h / 2)

            error = np.sqrt((gt_center[0] - track_center[0]) ** 2 + (gt_center[1] - track_center[1]) ** 2)
            center_errors.append(error)

        avg_error = np.mean(center_errors)
        print(f"Average center error for {seq['name']}: {avg_error:.2f} pixels")


if __name__ == "__main__":
    main()