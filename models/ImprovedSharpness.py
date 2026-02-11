import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# -------------------------- 1. 清晰度计算核心函数 --------------------------
def calculate_clarity_scores(image_paths):
    """
    计算一组图像的清晰度值（原始值+归一化值）
    :param image_paths: 图像路径列表（按slice_1到slice_7排序）
    :return: 原始清晰度值列表, 归一化清晰度值列表
    """
    # 定义清晰度评价卷积核
    # EOG
    # g = np.array([[0, 0, 0],
    #               [0, -1, 1],
    #               [0, 0, 0]])
    # t = np.array([[0, 0, 0],
    #               [0, -1, 0],
    #               [0, 1, 0]])
    #Roberts
    # g = np.array([[0, 0, 0],
    #               [0, -1, 0],
    #               [0, 0, 1]])
    # t = np.array([[0, 0, 0],
    #               [0, 0, 1],
    #               [0, -1, 0]])
    # Brenner
    # g = np.array([[0, 0, 0],
    #               [-1, 0, 1],
    #               [0, 0, 0]])
    # Tenengrad
    g = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
    t = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
    # Laplace
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]])
    E = []  # 原始清晰度值
    valid_paths = []  # 有效图像路径

    # 遍历图像计算清晰度
    for img_path in image_paths:
        # 读取灰度图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"警告：无法读取图像 {img_path}，赋值清晰度值为0")
            E.append(0.0)
            valid_paths.append(img_path)
            continue

        # 卷积计算x、y方向响应
        x = cv2.filter2D(img, -1, g)
        y = cv2.filter2D(img, -1, t)

        # 计算绝对值和作为清晰度特征
        # data = np.abs(x) + np.abs(y)
        # data = np.abs(x)
        # data = abs(cv2.filter2D(img, -1, kernel))
        data = abs(cv2.Laplacian(img, cv2.CV_16S))

        # 计算平均清晰度值（归一化到像素总数）
        e = data.sum() / (img.shape[0] * img.shape[1])
        E.append(e)
        valid_paths.append(img_path)

    # 归一化清晰度值到[0,1]
    E_np = np.array(E)
    maxvalue = E_np.max()
    minvalue = E_np.min()
    inputrange = maxvalue - minvalue

    if inputrange != 0:
        alpha = 1.0 / inputrange
        beta = -minvalue * alpha
        y_EOG = E_np * alpha + beta
    else:
        y_EOG = np.zeros_like(E_np)

    return E, y_EOG.tolist()


# -------------------------- 4. 主评估函数 --------------------------
def evaluate_and_plot(images_folder, labels_folder):
    # -------------------------- 加载真实标签 --------------------------
    labels = {}
    for i in range(1, 271):
        rank_file = os.path.join(labels_folder, f'ranks_{i}.txt')
        if not os.path.exists(rank_file):
            print(f"警告：未找到标签文件 {rank_file}，跳过该组")
            continue
        with open(rank_file, 'r') as f:
            # 读取文件内容，按逗号分隔并转换为整数
            ranks = list(map(int, f.read().split(',')))
        labels[i] = ranks

    # 汇总所有组的真实标签和预测标签
    all_true_labels = []
    all_predicted_labels = []

    # -------------------------- 遍历每组图像计算清晰度并排名 --------------------------
    for group_index in range(1, 271):
        # 跳过无标签的组
        if group_index not in labels:
            continue

        # 构建组文件夹路径
        group_folder = os.path.join(images_folder, f'group_{group_index}')
        if not os.path.exists(group_folder):
            print(f"警告：未找到图像文件夹 {group_folder}，跳过该组")
            continue

        # 构建7张图像的路径（slice_1到slice_7）
        image_paths = [os.path.join(group_folder, f'slice_{i}.jpg') for i in range(1, 8)]

        # 检查图像文件是否存在
        valid_image_paths = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                valid_image_paths.append(img_path)
            else:
                print(f"警告：未找到图像 {img_path}，该位置赋值清晰度值为0")
        # 补全7张图像（缺失的赋值为0）
        while len(valid_image_paths) < 7:
            valid_image_paths.append(None)

        # 计算清晰度值
        raw_clarity, norm_clarity = calculate_clarity_scores(image_paths)

        # 根据清晰度值生成预测排名（清晰度越高，排名越靠前）
        # 按清晰度降序排列，生成1-7的排名
        clarity_np = np.array(raw_clarity)
        sorted_indices = np.argsort(-clarity_np)  # 降序索引
        predicted_ranks = np.empty_like(sorted_indices)
        predicted_ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
        predicted_ranks = predicted_ranks.tolist()

        # 获取真实标签
        true_labels = labels[group_index]

        # 校验标签长度（确保是7个）
        if len(true_labels) != 7:
            print(f"警告：组 {group_index} 标签长度不为7，跳过该组")
            continue

        # 汇总标签
        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_ranks)

        # 可选：打印单组结果
        # print(f"组 {group_index} 真实排名：{true_labels}")
        # print(f"组 {group_index} 预测排名：{predicted_ranks}")

    # -------------------------- 计算混淆矩阵 --------------------------
    if not all_true_labels or not all_predicted_labels:
        print("错误：无有效标签数据，无法计算指标")
        return

    # 计算混淆矩阵（类别1-7）
    cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=range(1, 8))

    # -------------------------- 计算每个类别的指标 --------------------------
    metrics = {i: {} for i in range(1, 8)}
    total_samples = cm.sum()

    # 逐类别计算精确率、召回率、F1、准确率
    for i in range(7):  # 类别1-7对应索引0-6
        true_positive = cm[i, i]  # 真正例（对角线）
        predicted_positive = cm[:, i].sum()  # 预测为该类的总数
        actual_positive = cm[i, :].sum()  # 实际为该类的总数

        # 计算指标（避免除以0）
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0
        recall = true_positive / actual_positive if actual_positive > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positive / actual_positive if actual_positive > 0 else 0

        # 保存结果
        metrics[i + 1] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

    # -------------------------- 打印每个类别的指标 --------------------------
    print("=" * 60)
    print("每个类别的评估指标（基于清晰度排名）")
    print("=" * 60)
    for class_label, scores in metrics.items():
        print(f"类别 {class_label}:")
        print(f"  Accuracy:  {scores['Accuracy']:.4f}")
        print(f"  Precision: {scores['Precision']:.4f}")
        print(f"  Recall:    {scores['Recall']:.4f}")
        print(f"  F1 Score:  {scores['F1 Score']:.4f}")
        print()

    # -------------------------- 计算整体指标 --------------------------
    overall_metrics = calculate_metrics(all_true_labels, all_predicted_labels)

    # 打印整体指标
    print("=" * 60)
    print("整体评估指标（基于清晰度排名）")
    print("=" * 60)
    print(f"Accuracy:  {overall_metrics['Accuracy']:.4f}")
    print(f"Precision: {overall_metrics['Precision']:.4f}")
    print(f"Recall:    {overall_metrics['Recall']:.4f}")
    print(f"F1 Score:  {overall_metrics['F1 Score']:.4f}")
    print("=" * 60)

    # -------------------------- 绘制混淆矩阵 --------------------------
    plot_confusion_matrix(cm, save_path="confusion_matrix_clarity.png")


# -------------------------- 运行示例 --------------------------
if __name__ == "__main__":
    # 请替换为你的实际路径
    IMAGES_FOLDER = "./dataset/test/images"  # 包含group_1到group_270的文件夹
    LABELS_FOLDER = "./dataset/test/labels"  # 包含ranks_1到ranks_270.txt的文件夹

    # 执行评估
    evaluate_and_plot(IMAGES_FOLDER, LABELS_FOLDER)

