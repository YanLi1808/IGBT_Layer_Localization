import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告
import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from itertools import combinations
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import time
from torch.utils.data import ConcatDataset
import cv2
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import os
import glob

import os
import glob
from multimodels import get_binary_classification_model
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from torchvision import models
from utils import GradCAM, show_cam_on_image, center_crop_img

def load_groups_automatically(image_root="dataset", label_root="dataset"):
    """
    自动匹配图像组和对应的排名文件
    目录结构:
        dataset/
            train/
                images/
                    group_1/
                        img_0.jpg ... img_8.jpg
                    group_2/
                    ...
                labels/
                    ranks_1.txt  # 内容示例: 1,3,2,5,4,7,6,9,8
                    ranks_2.txt
                    ...
            test/
                images/
                    group_1/
                    ...
                labels/
                    ranks_1.txt
                    ...
    """
    groups_train = []
    groups_test = []

    # 获取训练集和测试集的图像组文件夹
    image_train_dirs = sorted(glob.glob(os.path.join(image_root, "train", "images", "group_*")))
    image_test_dirs = sorted(glob.glob(os.path.join(image_root, "test", "images", "group_*")))

    # 加载训练集数据
    for img_dir in image_train_dirs:
        # 提取group编号 (例如从 "dataset/train/images/group_3" 中提取3)
        group_num = os.path.basename(img_dir).split("_")[-1]

        # 匹配对应的标签文件
        label_path = os.path.join(label_root, "train", "labels", f"ranks_{group_num}.txt")

        if not os.path.exists(label_path):
            print(f"Warning: Missing label file for {img_dir}, skipping...")
            continue

        # 读取排名标签
        with open(label_path, 'r') as f:
            raw_ranks = f.read().strip()
            try:
                ranks = list(map(int, raw_ranks.split(',')))
                if len(ranks) != 7:
                    raise ValueError
            except:
                print(f"Invalid label format in {label_path}, skipping...")
                continue

        groups_train.append({
            "path": img_dir,
            "ranks": ranks
        })

    # 加载测试集数据
    for img_dir in image_test_dirs:
        # 提取group编号 (例如从 "dataset/test/images/group_3" 中提取3)
        group_num = os.path.basename(img_dir).split("_")[-1]

        # 匹配对应的标签文件
        label_path = os.path.join(label_root, "test", "labels", f"ranks_{group_num}.txt")

        if not os.path.exists(label_path):
            print(f"Warning: Missing label file for {img_dir}, skipping...")
            continue

        # 读取排名标签
        with open(label_path, 'r') as f:
            raw_ranks = f.read().strip()
            try:
                ranks = list(map(int, raw_ranks.split(',')))
                if len(ranks) != 7:
                    raise ValueError
            except:
                print(f"Invalid label format in {label_path}, skipping...")
                continue

        groups_test.append({
            "path": img_dir,
            "ranks": ranks
        })

    return groups_train, groups_test



# -------------------- 1. 数据预处理与标签生成 --------------------
def generate_labels_from_ranks(rank_list):
    """根据专家标注的排名生成成对标签
    Args:
        rank_list (list): 长度为9的列表，值越小表示质量越高(如rank=1为最佳)
    Returns:
        torch.Tensor: 形状为[36]的标签张量，1表示i应排在j前，-1反之
    """
    labels = []
    for i, j in combinations(range(7), 2):
        if rank_list[i] < rank_list[j]:
            labels.append(1)
        else:
            labels.append(-1)
    return torch.tensor(labels, dtype=torch.float32)


# 示例：假设某组的专家标注排名为 [1, 3, 2, 5, 4, 7, 6, 9, 8]
# 注意：值越小表示质量越高，即rank=1为最佳图像
# example_ranks = [1, 3, 2, 5, 4, 7, 6, 9, 8]
# group_labels = generate_labels_from_ranks(example_ranks)

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)

        img -= img.mean()
        img /= img.std()

        return img


class RandomHorizontalFlip(object):
    def __call__(self, img):

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT) #Image.FLIP_TOP_BOTTOM上下翻转

        return img

class RandomScaleCrop(object):
    def __init__(self, crop_size, fill=0):
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, img):
        img = img.resize((552, 552), Image.BILINEAR)

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))#按照crop_size进行随机裁剪

        return img

# -------------------- 2. 数据集加载器 --------------------
class RankDataset(Dataset):
    def __init__(self, group_dir, rank_list, transform=None):
        """
        Args:
            group_dir (str): 包含9张图像的文件夹路径
            rank_list (list): 该组的排名列表(长度9)
            transform: 图像预处理
        """
        self.group_dir = group_dir
        self.transform = transform or transforms.Compose([

            transforms.Resize((512, 512)),
            # RandomScaleCrop(crop_size=512),
            # RandomHorizontalFlip(),
            # RandomGaussianBlur(),
            Normalize(mean=0.485, std=0.229),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 加载所有图像
        self.images = []
        for i in range(7):
            img_path = os.path.join(group_dir, f"slice_{i+1}.jpg")
            img = Image.open(img_path).convert("RGB")
            self.images.append(self.transform(img))
        self.images = torch.stack(self.images)  # [9, C, H, W]

        # 生成标签
        self.labels = generate_labels_from_ranks(rank_list)
        self.pairs = list(combinations(range(7), 2))

    def __len__(self):
        return len(self.pairs)  # 36对

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.images[i], self.images[j], self.labels[idx]

# -------------------- 3. 模型定义 --------------------
class SiameseRanker_multimodels(nn.Module):
    def __init__(self):
        super().__init__()
        # 可以替换为其他模型名称
        model_names = ["alexnet", "vgg11", "vgg16", "vgg19",
                       "resnet18", "resnet50", "googlenet",
                       "mobilenetv2", "efficientnet-b0", "densenet121",
                       "shufflenet_v2_x1_0", "squeezenet1_1"]

        # 测试加载第一个模型
        test_model_name = model_names[1]
        self.base_cnn = get_binary_classification_model(test_model_name, pretrained=True)

    def forward_pair(self, x1, x2):
        return self.base_cnn(x1), self.base_cnn(x2)

    def forward_single(self, x):
        return self.base_cnn(x)

# -------------------- 4. 带标签的Hinge Loss --------------------
class PairwiseHingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, s1, s2, labels):
        """计算考虑标签方向的损失
        Args:
            s1: 图像1的预测分数 [batch_size, 1]
            s2: 图像2的预测分数 [batch_size, 1]
            labels: 标签 (1或-1) [batch_size]
        """
        diff = (s1 - s2).squeeze(1)  # [batch_size]
        loss = torch.relu(self.margin - labels * diff)
        return loss.mean()


# -------------------- 5. 训练流程 --------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    iter = 0
    total_loss = 0
    correct_count = 0
    total_count = 0
    for img1, img2, labels in dataloader:
        iter += 1
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()
        s1, s2 = model.forward_pair(img1, img2)
        loss = criterion(s1, s2, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        labels = labels.view(-1, 1)

        correct_count += ((s1 - s2) * labels > 0).sum().item()
        total_count += labels.size(0)

    accuracy = correct_count / total_count
    print(f"Iteration: {iter}, TrainAccuracy: {accuracy * 100:.2f}%")
    return total_loss / len(dataloader), accuracy

def test_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_count = 0
    iter = 0
    total_count = 0
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            iter += 1
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            s1, s2 = model.forward_pair(img1, img2)
            loss = criterion(s1, s2, labels)

            total_loss += loss.item()

            labels = labels.view(-1, 1)

            correct_count += ((s1 - s2) * labels > 0).sum().item()
            total_count += labels.size(0)

    accuracy = correct_count / total_count
    print(f"Iteration: {iter}, TestAccuracy: {accuracy * 100:.2f}%")
    return total_loss / len(dataloader), accuracy


def set_seed(seed=42):
    """设置随机种子确保可复现性"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_memory_usage():
    """
    获取当前GPU的显存使用情况
    返回：已用显存(MB)、总显存(MB)、剩余显存(MB)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0  # 无GPU时返回0
    # 获取当前设备（默认第0块GPU）
    device = torch.cuda.current_device()
    # 总显存
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2  # 转换为MB
    # 已用显存
    used_memory = torch.cuda.memory_allocated(device) / 1024**2
    # 剩余显存 = 总显存 - 已用显存（近似值）
    free_memory = total_memory - used_memory
    return round(used_memory, 2), round(total_memory, 2), round(free_memory, 2)


# --------------------------------------------------------------模型训练---------------------------------------------------------------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 16

    # 假设数据存储在以下结构中：
    # dataset/
    #   ├── group_1/
    #   │   ├── img_0.jpg
    #   │   └── ...
    #   ├── group_2/
    #   │   ├── img_0.jpg
    #   │   └── ...
    #   └── ...

    # 加载数据
    train_groups, test_groups = load_groups_automatically()

    # 创建数据集
    train_dataset = []
    for group in train_groups:
        dataset = RankDataset(
            group_dir=group["path"],
            rank_list=group["ranks"]
        )
        train_dataset.append(dataset)

    test_dataset = []
    for group in test_groups:
        dataset = RankDataset(
            group_dir=group["path"],
            rank_list=group["ranks"]
        )
        test_dataset.append(dataset)

    # 合并数据集

    combined_train_dataset = ConcatDataset(train_dataset)
    combined_test_dataset = ConcatDataset(test_dataset)

    # 创建 DataLoader
    train_dataloader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    test_dataloader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)

    # 初始化模型
    model = SiameseRanker_multimodels().to(device)
    criterion = PairwiseHingeLoss(margin=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # 损失列表
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0

    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # 创建文件夹，使用当前时间戳作为文件夹名称
    save_dir = f"weights/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    # 使用 StepLR 调度器，每50个epoch学习率减少十分之一
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    # 训练循环
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_accuracy = train_epoch(model, train_dataloader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 测试
        test_loss, test_accuracy = test_epoch(model, test_dataloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # 保存模型，命名加上当前epoch数和损失值
            model_filename = os.path.join(save_dir,
                                          f"rank_model_epoch{epoch + 1}_train_loss_{train_loss:.4f}_test_loss_{test_loss:.4f}_train_accuracy_{train_accuracy * 100:.2f}%_test_accuracy_{test_accuracy * 100:.2f}%.pth")
            torch.save(model.state_dict(), model_filename)

        # 每个epoch结束时更新学习率
        scheduler.step()  # 更新学习率

    last_model = os.path.join(save_dir, f"rank_model_epoch{epoch + 1}_train_loss_{train_loss:.4f}_test_loss_{test_loss:.4f}_train_accuracy_{train_accuracy * 100:.2f}%_test_accuracy_{test_accuracy * 100:.2f}%.pth")
    torch.save(model.state_dict(), last_model)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", color='blue', marker='o')
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss", color='red', marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")  # 保存到本地文件
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy", color='blue', marker='o')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy", color='red', marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve.png")  # 保存到本地文件
    plt.show()



# --------------------------------------------------------------模型测试---------------------------------------------------------------------------
transform = transforms.Compose([
        transforms.Resize((512, 512)),

        Normalize(mean=0.485, std=0.229),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# 只有在第一次调用时加载模型
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseRanker_multimodels().to(device)
    model.load_state_dict(torch.load("./weights/VGG11/pretrained/rank_model_epoch49_train_loss_0.0006_test_loss_0.0091_train_accuracy_100.00%_test_accuracy_99.14%.pth"))
    model.eval()
    model.eval()
    return model, device


def predict_batch(image_paths, model, device):
    """批量预测"""
    scores = []
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            score = model.forward_single(img)
        scores.append(score.item())
    return scores

# --------------------------------------------------------------1、Top1排序---------------------------------------------------------------------------
def evaluate_predictions(images_folder, labels_folder):
    correct_predictions = 0
    total_predictions = 0
    results = []

    # 加载模型
    model, device = load_model()

    # 使用ThreadPoolExecutor实现并行化
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        # 使用 tqdm 包装 group_idx 循环，添加进度条
        for group_idx in tqdm(range(1, 271), desc="Processing groups", ncols=100):
            group_folder = os.path.join(images_folder, f"group_{group_idx}")
            label_file = os.path.join(labels_folder, f"ranks_{group_idx}.txt")

            # 获取每个文件夹下的7张图片路径
            image_paths = [os.path.join(group_folder, f"slice_{slice_idx}.jpg") for slice_idx in range(1, 8)]

            # 异步提交任务
            futures.append(executor.submit(predict_batch, image_paths, model, device))

            # 获取 ranks 文件的排序信息
            with open(label_file, 'r') as f:
                ranks = []
                for line in f.readlines():
                    ranks.extend([int(x) for x in line.strip().split(',')])

            # 等待任务完成并获取结果
            future = futures.pop(0)
            image_scores = future.result()

            # 找到得分最高的图像
            highest_score_idx = np.argmax(image_scores)
            results.append(ranks[highest_score_idx])

            # 如果最高得分的图像在 ranks 中的排名是 1，则预测正确
            if ranks[highest_score_idx] in [1]:
                correct_predictions += 1

            total_predictions += 1

    # 打印最终的预测准确率
    accuracy = correct_predictions / total_predictions
    print(f"Correct predictions: {correct_predictions}, Total predictions: {total_predictions}")
    print(f"Prediction accuracy: {accuracy * 100:.2f}%")

# -----------------------------------------------------------------2、绘制混淆矩阵----------------------------------------------------------------------
def plot_confusion_matrix(cm, save_path=None, figsize=(8, 6), dpi=300):
    """
    绘制并可选保存混淆矩阵

    参数:
        cm: 混淆矩阵数组
        save_path: 保存路径（如"output/confusion_matrix.png"），为None时不保存
        figsize: 图片尺寸
        dpi: 保存图片的分辨率
    """
    # 绘制混淆矩阵
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(1, 8),
                yticklabels=range(1, 8))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Rank')
    plt.ylabel('True Rank')

    # 保存图片（如果指定了路径）
    if save_path is not None:
        # 确保保存目录存在
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # 保存图片
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {os.path.abspath(save_path)}")

    # 显示图片
    plt.show()
    # 关闭当前图形，避免内存占用
    plt.close()


def calculate_metrics(true_labels, predicted_labels):
    """
    计算 Accuracy, Precision, Recall 和 F1-Score
    :param true_labels: 真实标签
    :param predicted_labels: 预测标签
    :return: 各项评估指标的字典
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    return metrics

# 结合加载模型、批量预测及混淆矩阵计算
def evaluate_and_plot(images_folder, labels_folder):
    # 加载模型
    model, device = load_model()

    # 加载标签（真实标签）
    labels = {}
    for i in range(1, 271):
        rank_file = os.path.join(labels_folder, f'ranks_{i}.txt')
        with open(rank_file, 'r') as f:
            # 读取文件内容，按逗号分隔并将每个部分转换为整数
            ranks = list(map(int, f.read().split(',')))

        labels[i] = ranks

    # 汇总所有组的真实标签和预测标签
    all_true_labels = []
    all_predicted_labels = []


    for group_index in range(1, 271):
        # 读取每个组的图像路径并进行预测
        group_folder = os.path.join(images_folder, f'group_{group_index}')
        image_paths = [os.path.join(group_folder, f'slice_{i}.jpg') for i in range(1, 8)]  # 假设每个组有7张图片

        # 获取预测得分
        predicted_scores = predict_batch(image_paths, model, device)

        # 获取真实标签
        true_labels = labels[group_index]

        # 排序预测得分并获取预测排名
        # 计算排名
        sorted_indices = np.argsort(-np.array(predicted_scores))  # 按降序排列的索引
        predicted_ranks = np.empty_like(sorted_indices)
        predicted_ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)


        # 将真实标签和预测标签添加到列表
        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_ranks)


    # 计算混淆矩阵
    cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=range(1, 8))



    # 初始化结果字典
    metrics = {i: {} for i in range(1, 8)}

    # 计算总样本数
    total_samples = cm.sum()

    # 计算每个类别的指标
    for i in range(7):  # 类别1到7对应索引0到6
        true_positive = cm[i, i]  # 对角线上的值

        # 计算精确率 (Precision)
        predicted_positive = cm[:, i].sum()  # 第i列的和
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0

        # 计算召回率 (Recall)
        actual_positive = cm[i, :].sum()  # 第i行的和
        recall = true_positive / actual_positive if actual_positive > 0 else 0

        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 计算准确率 (Accuracy) 对于单个类别（通常不常用，这里计算的是该类别样本的分类准确率）
        accuracy = true_positive / actual_positive if actual_positive > 0 else 0

        # 保存结果
        metrics[i + 1] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }


    # 打印结果
    for class_label, scores in metrics.items():
        print(f"类别 {class_label}:")
        print(f"  Accuracy:  {scores['Accuracy']:.4f}")
        print(f"  Precision: {scores['Precision']:.4f}")
        print(f"  Recall:    {scores['Recall']:.4f}")
        print(f"  F1 Score:  {scores['F1 Score']:.4f}")
        print()



    # 计算各项指标的平均值
    metrics = calculate_metrics(all_true_labels, all_predicted_labels)

    # 输出计算结果
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, save_path="confusion_matrix.png")

# -----------------------------------------------------------------3、计算分类指标（准确率,召回率,F1)----------------------------------------------------------------------
def metrics():

    # 加载数据并添加进度条
    print("Loading groups...")
    train_groups, test_groups = load_groups_automatically()

    # 创建数据集 - 使用列表推导式优化
    print("Creating datasets...")

    test_dataset = [
        RankDataset(group_dir=group["path"], rank_list=group["ranks"])
        for group in tqdm(test_groups, desc="Creating test datasets", ncols=100)
    ]

    # 合并数据集
    combined_test_dataset = ConcatDataset(test_dataset)

    # 优化DataLoader参数，可能提升性能
    test_dataloader = DataLoader(
        combined_test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,  # 启用pin_memory加速GPU传输
        prefetch_factor=2  # 预加载数据
    )

    # 加载模型
    model, device = load_model()
    model.eval()

    # 存储所有样本的预测结果和真实标签
    all_preds = []
    all_targets = []
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        # 添加测试过程的进度条
        for img1, img2, labels in tqdm(
                test_dataloader,
                desc="Testing",
                total=len(test_dataloader),
                ncols=100
        ):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # 模型预测
            s1, s2 = model.forward_pair(img1, img2)
            labels = labels.view(-1, 1)

            # 计算预测结果和正确性
            preds = ((s1 - s2) > 0).float().cpu().numpy()
            targets = (labels == 1).float().cpu().numpy()

            # 收集所有批次的结果
            all_preds.extend(preds)
            all_targets.extend(targets)

            # 计算正确预测数
            correct_count += ((s1 - s2) * labels > 0).sum().item()
            total_count += labels.size(0)

    # 计算准确率
    accuracy = correct_count / total_count
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 转换为numpy数组便于计算指标
    all_preds_np = np.array(all_preds).flatten()
    all_targets_np = np.array(all_targets).flatten()

    # 计算评价指标
    accuracy = accuracy_score(all_targets_np, all_preds_np)
    precision = precision_score(all_targets_np, all_preds_np, zero_division=1)
    recall = recall_score(all_targets_np, all_preds_np, zero_division=1)
    f1 = f1_score(all_targets_np, all_preds_np, zero_division=1)

    # 打印结果
    print(f"Total samples: {len(all_targets_np)}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


# -----------------------------------------------------------------4、绘制热力图----------------------------------------------------------------------
def process_image(model, device, img_path, target_layers, save_path):
    """处理单张图片并保存可视化结果"""
    try:
        # 数据转换
        data_transform = transforms.Compose([

            transforms.Resize((512, 512)),
            Normalize(mean=0.485, std=0.229),
            transforms.ToTensor()
        ])

        # 加载图片
        img = Image.open(img_path).convert('RGB')
        img_tensor = data_transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

        # 初始化GradCAM
        use_cuda = device.type == "cuda"
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        target_category = 0  # 根据您的需求调整

        # 计算GradCAM
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]

        # 准备可视化
        img = img.resize((512, 512))
        img_np = np.array(img, dtype=np.uint8)
        visualization = show_cam_on_image(
            img_np.astype(dtype=np.float32) / 255.,
            grayscale_cam,
            use_rgb=True
        )

        # 保存结果
        plt.figure(figsize=(10, 10))
        plt.imshow(visualization)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭图像以释放内存

        return True
    except Exception as e:
        print(f"处理图片 {img_path} 时出错: {str(e)}")
        return False


def batch_process():
    # 加载模型
    model, device = load_model()
    if model is None or device is None:
        print("模型加载失败")
        return

    # 设置目标层 - 根据您的模型调整
    # target_layers = [model.base_cnn.layer4]
    # target_layers = [model.base_cnn.stage4]
    target_layers = [model.base_cnn.features[-2]]

    # 定义路径
    root_dir = "dataset/train/images"  # 源图片根目录 test:18;train:40
    output_root = "grad_cam_results/train"  # 结果保存根目录

    # 创建结果根目录
    os.makedirs(output_root, exist_ok=True)

    # 记录处理统计
    total_success = 0
    total_failed = 0

    # 遍历所有group文件夹 (1到270)
    for group in range(1, 41):
        group_name = f"group_{group}"
        group_dir = os.path.join(root_dir, group_name)
        output_group_dir = os.path.join(output_root, group_name)

        # 创建对应的结果文件夹
        os.makedirs(output_group_dir, exist_ok=True)

        # 检查源文件夹是否存在
        if not os.path.exists(group_dir):
            print(f"警告: 源文件夹 {group_dir} 不存在，跳过该组")
            continue

        # 处理该组中的7张图片 (slice_1到slice_7)
        group_success = 0
        group_failed = 0

        for slice_num in range(1, 8):
            img_file = f"slice_{slice_num}.jpg"  # 假设都是jpg格式
            img_path = os.path.join(group_dir, img_file)

            # 检查图片文件是否存在
            if not os.path.exists(img_path):
                print(f"警告: 图片 {img_path} 不存在，跳过")
                total_failed += 1
                group_failed += 1
                continue

            # 构建保存路径
            save_file = f"slice_{slice_num}_grad_cam.jpg"
            save_path = os.path.join(output_group_dir, save_file)

            # 处理图片
            success = process_image(model, device, img_path, target_layers, save_path)

            if success:
                group_success += 1
                total_success += 1
            else:
                group_failed += 1
                total_failed += 1

        # 打印每组处理结果
        print(f"组 {group_name} 处理完成 - 成功: {group_success}, 失败: {group_failed}")

    # 打印总体处理结果
    print("\n批量处理全部完成")
    print(f"总处理图片数: {total_success + total_failed}")
    print(f"成功: {total_success}, 失败: {total_failed}")
    print(f"成功率: {total_success / (total_success + total_failed) * 100:.2f}%")


# 调用测试函数进行排序
if __name__ == "__main__":
    # 训练
    # main()
    # 测试
    images_folder = "dataset/test/images"  # 替换为你 images 文件夹的路径
    labels_folder = "dataset/test/labels"  # 替换为你 labels 文件夹的路径
    # 1、TOP-1排序
    evaluate_predictions(images_folder, labels_folder)
    # 2、绘制混淆矩阵
    # evaluate_and_plot(images_folder, labels_folder)
    # 3、计算分类指标
    # metrics()
    # 4、绘制热力图
    # batch_process()