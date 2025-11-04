from src.xPatch import xPatch
from src.RNN import RNN


MODEL_REGISTRY = {
    'xpatch': xPatch,
    'rnn': RNN,
}


def get_model_class(name: str):
    """
    根据模型名称字符串从注册表中获取模型类。
    """
    model_class = MODEL_REGISTRY.get(name.lower())
    if model_class is None:
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return model_class
