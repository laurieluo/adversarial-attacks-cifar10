from src.models import ResNet18, VGG16_BN, DenseNet121
from src.wideresnet import WideResNet28_10, WideResNet94_16


def print_model_layers(model, model_name):
    """
    打印模型的所有层名称

    Args:
        model: 模型实例
        model_name: 模型名称
    """
    print(f"层名称列表（{model_name}）：")
    for name, module in model.named_modules():
        if name:  # 过滤掉根模块（空名称）
            print(name)
    print()  # 添加空行分隔


if __name__ == "__main__":
    # 定义模型字典
    models = {
        'resnet18': ResNet18(),
        'vgg16': VGG16_BN(),
        'densenet121': DenseNet121(),
        'wrn2810': WideResNet28_10(),
        'wrn9416': WideResNet94_16()
    }

    # 遍历打印所有模型的层名称
    for name, model in models.items():
        print_model_layers(model, name)
