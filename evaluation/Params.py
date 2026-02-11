import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from torchprofile import profile_macs

# 假设这些函数已经定义，用于获取GPU内存使用情况
def get_gpu_memory_usage():
    """获取当前GPU的内存使用情况"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        used = torch.cuda.memory_allocated(0) // (1024 * 1024)
        free = total - used
        return used, total, free
    return 0, 0, 0


def count_model_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_vgg_official_layers(model):
    """统计VGG模型的卷积层和全连接层数量（适用于官方实现）"""
    layer_count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_count += 1
    return layer_count


def get_binary_classification_model(model_name, pretrained=True):
    """获取适用于二分类任务的模型"""
    model_func = getattr(models, model_name, None)
    if not model_func:
        raise ValueError(f"模型 {model_name} 不存在")

    model = model_func(pretrained=pretrained)

    # 根据不同模型修改最后一层以适应二分类
    if model_name.startswith('vgg'):
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1)
    elif model_name.startswith('resnet'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
    elif model_name.startswith('googlenet'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
    elif model_name.startswith('mobilenet'):
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1)
    elif model_name.startswith('efficientnet'):
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1)
    elif model_name.startswith('densenet'):
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1)
    elif model_name.startswith('shufflenet'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
    elif model_name.startswith('squeezenet'):
        model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == 'alexnet':
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1)

    return model


def calculate_flops(model, input_size):
    """计算模型的FLOPs（浮点运算次数）"""


    # 创建一个虚拟输入用于计算
    dummy_input = torch.randn(1, 3, input_size, input_size).to(next(model.parameters()).device)

    # 计算MACs (Multiply-Accumulate Operations)
    macs = profile_macs(model, dummy_input)

    # 通常认为1 MAC = 2 FLOPs (一次乘法和一次加法)
    flops = macs * 2

    return flops


if __name__ == "__main__":
    model_names = ["alexnet", "vgg11", "vgg16", "vgg19",
                   "resnet18", "resnet50", "googlenet",
                   "mobilenetv2", "efficientnet-b0", "densenet121",
                   "shufflenet_v2_x1_0", "squeezenet1_1"]

    # 可修改的参数
    test_model_index = 11  # 模型索引
    batch_size = 1  # 在这里调整batch_size大小
    input_size = 512  # 输入图像尺寸

    test_model_name = model_names[test_model_index]

    # 初始显存状态
    print("=== 初始显存状态 ===")
    used, total, free = get_gpu_memory_usage()
    print(f"GPU显存: 已用 {used} MB / 总 {total} MB (剩余 {free} MB)")

    # 加载模型
    model = get_binary_classification_model(test_model_name, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\n模型 {test_model_name} 已加载至 {device} (batch_size={batch_size})")

    # 模型加载后显存
    print("\n=== 模型加载后显存状态 ===")
    used, total, free = get_gpu_memory_usage()
    print(f"GPU显存: 已用 {used} MB / 总 {total} MB (剩余 {free} MB)")

    # 模型信息
    layer_count = count_vgg_official_layers(model)
    total_params, trainable_params = count_model_parameters(model)
    print(f"\n模型信息统计:")
    print(f"卷积+全连接层数: {layer_count}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 计算FLOPs
    flops = calculate_flops(model, input_size)
    print(f"FLOPs: {flops:.2e} (约 {flops / 1e9:.2f} G)")

    # 模拟训练（使用指定batch_size）
    print("\n=== 模拟训练过程 ===")
    # 生成指定batch_size的输入数据
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    dummy_label = torch.rand(batch_size, 1).to(device)  # 随机标签
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 前向传播
    model.train()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_label)
    print(f"损失值: {loss.item():.6f}")

    # 前向传播后显存（受batch_size影响较大）
    used_forward, _, _ = get_gpu_memory_usage()
    print(f"前向传播后显存使用: {used_forward} MB (batch_size={batch_size})")

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 反向传播后显存（batch_size越大，梯度占用显存越多）
    used_backward, _, _ = get_gpu_memory_usage()
    print(f"反向传播后显存使用: {used_backward} MB (batch_size={batch_size})")

    # 输出信息
    print(f"\n测试输入: batch_size={batch_size}, 形状={dummy_input.shape}")
    print(f"模型输出形状: {outputs.shape}")
