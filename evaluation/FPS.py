import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from torchprofile import profile_macs
import time
import gc


# 获取GPU内存使用情况（清理缓存后）
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        used = torch.cuda.memory_allocated(0) // (1024 * 1024)
        free = total - used
        return used, total, free
    return 0, 0, 0


# 计算模型参数量
def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# 统计卷积+全连接层数
def count_vgg_official_layers(model):
    layer_count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_count += 1
    return layer_count


# 修复：适配所有torchvision模型的加载逻辑
def get_binary_classification_model(model_name, pretrained=True):
    """获取适用于二分类任务的模型（修复module not callable问题）"""
    # 定义模型映射：处理特殊模型的加载方式
    model_mapping = {
        # 基础模型（直接可调用）
        "alexnet": models.alexnet,
        "vgg11": models.vgg11,
        "vgg16": models.vgg16,
        "vgg19": models.vgg19,
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "googlenet": models.googlenet,
        "mobilenetv2": models.mobilenet_v2,  # 注意：原名称是mobilenet_v2而非mobilenetv2
        "efficientnet-b0": models.efficientnet_b0,  # 注意：原名称是efficientnet_b0
        "densenet121": models.densenet121,
        "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
        "squeezenet1_1": models.squeezenet1_1
    }

    # 检查模型名称是否合法
    if model_name not in model_mapping:
        raise ValueError(f"不支持的模型 {model_name}，支持列表：{list(model_mapping.keys())}")

    # 加载模型（核心修复：使用映射后的可调用函数）
    model_func = model_mapping[model_name]
    model = model_func(pretrained=pretrained)

    # 根据不同模型修改最后一层适配二分类
    if model_name.startswith('vgg'):
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1)
    elif model_name.startswith('resnet'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
    elif model_name == 'googlenet':
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
    elif model_name == 'mobilenetv2':
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1)
    elif model_name == 'efficientnet-b0':
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


# 计算FLOPs
def calculate_flops(model, input_size):
    dummy_input = torch.randn(1, 3, input_size, input_size).to(next(model.parameters()).device)
    macs = profile_macs(model, dummy_input)
    flops = macs * 2
    return flops


# 计算推理速度（FPS）
def calculate_inference_speed(model, input_size, batch_size=1, warmup=10, repeat=100):
    device = next(model.parameters()).device
    model.eval()
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # CUDA同步
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 计时推理
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    # 计算速度
    total_time = end_time - start_time
    total_samples = batch_size * repeat
    avg_fps = total_samples / total_time
    avg_latency = (total_time / repeat) * 1000

    return avg_fps, avg_latency


if __name__ == "__main__":
    # 修复：模型名称与torchvision官方命名对齐
    model_names = ["alexnet", "vgg11", "vgg16", "vgg19",
                   "resnet18", "resnet50", "googlenet",
                   "mobilenetv2", "efficientnet-b0", "densenet121",
                   "shufflenet_v2_x1_0", "squeezenet1_1"]

    # 可修改参数
    test_model_index = 6  # 测试squeezenet1_1
    batch_size = 1
    input_size = 512

    test_model_name = model_names[test_model_index]

    # 初始显存
    print("=== 初始显存状态 ===")
    used, total, free = get_gpu_memory_usage()
    print(f"GPU显存: 已用 {used} MB / 总 {total} MB (剩余 {free} MB)")

    # 加载模型（修复后可正常调用）
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

    # 推理速度测试
    print("\n=== 推理速度测试 ===")
    avg_fps, avg_latency = calculate_inference_speed(
        model, input_size, batch_size, warmup=10, repeat=1000
    )
    print(f"推理预热次数: 10, 重复测试次数: 100")
    print(f"平均推理延迟: {avg_latency:.2f} ms/次 (batch_size={batch_size})")
    print(f"平均FPS (每秒处理样本数): {avg_fps:.2f}")
    print(f"等效单张图像FPS: {avg_fps / batch_size:.2f}")

    # 模拟训练
    print("\n=== 模拟训练过程 ===")
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    dummy_label = torch.rand(batch_size, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_label)
    print(f"损失值: {loss.item():.6f}")

    # 前向传播后显存
    used_forward, _, _ = get_gpu_memory_usage()
    print(f"前向传播后显存使用: {used_forward} MB (batch_size={batch_size})")

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 反向传播后显存
    used_backward, _, _ = get_gpu_memory_usage()
    print(f"反向传播后显存使用: {used_backward} MB (batch_size={batch_size})")

    # 输出信息
    print(f"\n测试输入: batch_size={batch_size}, 形状={dummy_input.shape}")
    print(f"模型输出形状: {outputs.shape}")