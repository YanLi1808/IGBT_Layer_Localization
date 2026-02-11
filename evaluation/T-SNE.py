import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torchvision.transforms as transforms
import glob
from itertools import combinations
from main import SiameseRanker  # 导入你的孪生网络
from main import SiameseRanker_AlexNet
from main import SiameseRanker_multimodels

# -------------------- 1. 数据预处理与数据集（复用） --------------------
class Normalize(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img -= img.mean()
        img /= img.std() if img.std() != 0 else 1.
        return img


class RandomHorizontalFlip(object):
    def __call__(self, img):
        if np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class PairDataset(Dataset):
    def __init__(self, group_dir, rank_list, transform=None):
        self.group_dir = group_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            RandomHorizontalFlip(),
            Normalize(mean=0.485, std=0.229),
            transforms.ToTensor(),
        ])

        self.images = []
        for i in range(7):
            img_path = os.path.join(group_dir, f"slice_{i + 1}.jpg")
            img = Image.open(img_path).convert("RGB")
            self.images.append(self.transform(img))
        self.images = torch.stack(self.images)

        self.pairs = list(combinations(range(7), 2))
        self.labels = generate_labels_from_ranks(rank_list)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.images[i], self.images[j], self.labels[idx]


# -------------------- 2. 数据加载与工具函数（复用） --------------------
def generate_labels_from_ranks(rank_list):
    labels = []
    for i, j in combinations(range(7), 2):
        if rank_list[i] < rank_list[j]:
            labels.append(1)
        else:
            labels.append(-1)
    return torch.tensor(labels, dtype=torch.float32)


def load_pair_datasets(image_root="dataset", label_root="dataset"):
    """加载所有group的成对样本（img_i, img_j, 1/-1），返回训练集和测试集"""
    train_datasets = []
    test_datasets = []
    all_group_dirs = sorted(glob.glob(os.path.join(image_root, "*", "images", "group_*")))  # 匹配train和test的group

    for img_dir in all_group_dirs:
        group_num = os.path.basename(img_dir).split("_")[-1]
        split = "train" if "train" in img_dir else "test"
        label_path = os.path.join(label_root, split, "labels", f"ranks_{group_num}.txt")

        if not os.path.exists(label_path):
            print(f"Warning: Missing label for {img_dir}, skipping...")
            continue

        # 读取排名并生成成对标签
        with open(label_path, 'r') as f:
            raw_ranks = f.read().strip()
            try:
                ranks = list(map(int, raw_ranks.split(',')))
                if len(ranks) != 7:
                    raise ValueError(f"Expected 7 ranks, got {len(ranks)}")
            except:
                print(f"Invalid label in {label_path}, skipping...")
                continue

        dataset = PairDataset(img_dir, ranks)
        if split == "train":
            train_datasets.append(dataset)
        else:
            test_datasets.append(dataset)

    # 分别合并训练集和测试集
    train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    test_dataset = ConcatDataset(test_datasets) if test_datasets else None

    return train_dataset, test_dataset  # 返回训练集和测试集


# -------------------- 3. 特征提取（输入层 + 倒数第二层） --------------------
def extract_input_features(dataloader):
    """提取输入层特征（原始像素，展平拼接）"""
    input_features = []
    labels_list = []
    for img_i, img_j, label in dataloader:
        img_i_flat = img_i.numpy().reshape(img_i.shape[0], -1)
        img_j_flat = img_j.numpy().reshape(img_j.shape[0], -1)
        pair_feat = np.concatenate([img_i_flat, img_j_flat], axis=1)
        input_features.append(pair_feat)
        labels_list.append(label.numpy())
    return np.concatenate(input_features, axis=0), np.concatenate(labels_list, axis=0)


def extract_penultimate_features(model, dataloader, device):
    """提取倒数第二层特征（300维，拼接后600维）"""
    model.eval()
    features_list = []
    labels_list = []
    feature_cache = {"i": None, "j": None}

    def hook_fn(module, input, output):
        if feature_cache["i"] is None:
            feature_cache["i"] = output.detach().cpu().numpy()
        else:
            feature_cache["j"] = output.detach().cpu().numpy()

    # 倒数第二层：classifier[4]（ReLU输出，300维）
    target_layer = model.base_cnn.classifier[5]
    hook = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for img_i, img_j, label in dataloader:
            img_i, img_j = img_i.to(device), img_j.to(device)

            # 提取img_i特征
            feature_cache["i"] = None
            _ = model.forward_single(img_i)
            feat_i = feature_cache["i"]

            # 提取img_j特征
            feature_cache["j"] = None
            _ = model.forward_single(img_j)
            feat_j = feature_cache["j"]

            # 拼接特征
            pair_feat = np.concatenate([feat_i, feat_j], axis=1)  # 600维
            features_list.append(pair_feat)
            labels_list.append(label.numpy())

    hook.remove()
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


# -------------------- 4. 合并绘制t-SNE图（对比展示） --------------------

def plot_combined_tsne(input_features, penultimate_features, labels, save_path="combined_tsne.png"):
    """
    合并绘制输入层和倒数第二层的t-SNE图，左右对比，采用SCI顶刊风格
    """
    # 统一标签筛选（1/-1）
    mask_1 = (labels == 1)
    mask_neg1 = (labels == -1)

    # 设置顶顶刊图表样式配置
    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimHei"],  # 英文用Times New Roman，中文用黑体
        "axes.unicode_minus": False,  # 解决负号显示问题
        "axes.labelsize": 14,  # 坐标轴标签字体大小
        "axes.titlesize": 16,  # 标题字体大小
        "xtick.labelsize": 12,  # x轴刻度字体大小
        "ytick.labelsize": 12,  # y轴刻度字体大小
        "legend.fontsize": 12,  # 图例字体大小
        "lines.markersize": 6,  # 标记大小
        "figure.dpi": 300,  # 图像分辨率
        "axes.linewidth": 1.0  # 坐标轴线条宽度
    })

    # 输入层特征降维（先PCA再t-SNE）
    print("输入层特征降降维中...")
    pca_input = PCA(n_components=50, random_state=42)
    input_pca = pca_input.fit_transform(input_features)
    tsne_input = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(input_pca)

    # 倒数第二层特征降维
    print("倒数第二层特征降维中...")
    tsne_penultimate = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(penultimate_features)

    # 顶刊风格配色（使用更专业的颜色组合）
    colors = {
        'positive': '#2c7fb8',  # 蓝色系（代表正样本）
        'negative': '#e41a1c',  # 红色系（代表负样本）
        'grid': '0.85'  # 浅灰色网格
    }

    # 合并绘图（1行2列子图，增加整体尺寸）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # 子图1：输入层
    ax1.scatter(tsne_input[mask_1, 0], tsne_input[mask_1, 1],
                c=colors['positive'], label='Label=1', s=25, alpha=0.7, edgecolors='none')
    ax1.scatter(tsne_input[mask_neg1, 0], tsne_input[mask_neg1, 1],
                c=colors['negative'], label='Label=-1', s=25, alpha=0.7, edgecolors='none')
    # ax1.set_title("The Input Layer", pad=15)
    ax1.set_xlabel("Axial 1")
    ax1.set_ylabel("Axial 2")
    ax1.grid(True, color=colors['grid'], linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)  # 去除上边框
    ax1.spines['right'].set_visible(False)  # 去除右边框

    # 子图2：倒数第二层
    ax2.scatter(tsne_penultimate[mask_1, 0], tsne_penultimate[mask_1, 1],
                c=colors['positive'], label='Label=1', s=25, alpha=0.7, edgecolors='none')
    ax2.scatter(tsne_penultimate[mask_neg1, 0], tsne_penultimate[mask_neg1, 1],
                c=colors['negative'], label='Label=-1', s=25, alpha=0.7, edgecolors='none')
    # ax2.set_title("The First FC Layer", pad=15)
    ax2.set_xlabel("Axial 1")
    ax2.set_ylabel("Axial 2")
    ax2.grid(True, color=colors['grid'], linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)  # 去除上边框
    ax2.spines['right'].set_visible(False)  # 去除右边框

    # 统一图例（放在图外右侧，更整洁）
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))

    # 保存与显示（使用更高质量参数）
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    print(f"合并t-SNE图已保存至 {save_path}")

# -------------------- 主函数 --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    # 1. 加载成对样本数据集
    print("加载数据集...")

    train_dataset, test_dataset = load_pair_datasets()
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"总样本数：{len(test_dataset)}（270组×21对）")

    # 2. 提取输入层特征
    print("提取输入层特征...")
    input_feat, labels = extract_input_features(dataloader)

    # 3. 加载模型并提取倒数第二层特征
    print("加载模型...")
    # model = SiameseRanker_AlexNet().to(device)
    model = SiameseRanker_multimodels().to(device)
    model.load_state_dict(torch.load(
        "./weights/VGG16/pretrained/rank_model_epoch100_train_loss_0.0000_test_loss_0.0110_train_accuracy_100.00%_test_accuracy_98.84%.pth",  # 替换为你的模型路径
        map_location=device
    ))
    model.eval()

    print("提取倒数第二层特征...")
    penultimate_feat, _ = extract_penultimate_features(model, dataloader, device)

    # 4. 合并绘制t-SNE图
    print("绘制合并t-SNE图...")
    plot_combined_tsne(input_feat, penultimate_feat, labels)


if __name__ == "__main__":
    main()