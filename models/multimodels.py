import torch
import torch.nn as nn
import torchvision.models as models


def count_vgg_official_layers(model):
    """按VGG官方标准统计层数（卷积+全连接层）"""
    layer_count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_count += 1
    return layer_count


def count_model_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    return total_params, trainable_params


def get_gpu_memory_usage():
    """
    获取当前GPU的显存使用情况
    返回：已用显存(MB)、总显存(MB)、剩余显存(MB)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0  # 无GPU时返回0
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    used_memory = torch.cuda.memory_allocated(device) / 1024 ** 2
    free_memory = total_memory - used_memory
    return round(used_memory, 2), round(total_memory, 2), round(free_memory, 2)


def get_binary_classification_model(model_name, pretrained=False):
    """加载模型并修改为二分类任务"""
    if model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif model_name in ["vgg11", "vgg16", "vgg19"]:
        model = getattr(models, model_name)(pretrained=pretrained)
    elif model_name in ["resnet18", "resnet50", "googlenet"]:
        model = getattr(models, model_name)(pretrained=pretrained)
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=pretrained)
    elif model_name == "efficientnet-b0":
        model = models.efficientnet_b0(pretrained=pretrained)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 修改输出层
    if model_name in ["alexnet", "vgg11", "vgg16", "vgg19"]:
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    elif model_name in ["resnet18", "resnet50", "googlenet", "shufflenet_v2_x1_0"]:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    elif model_name == "mobilenetv2":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    elif model_name == "efficientnet-b0":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    elif model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    elif model_name == "squeezenet1_1":
        model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid(),
            nn.Flatten()
        )
    return model


# 使用示例（可调整batch_size）
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
