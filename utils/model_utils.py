from src.model.xPatch import xPatch
from src.model.RNN import RNN
from src.model.LSTM import LSTM
from src.model.GRU import GRU
from src.model.MLP import MLP


MODEL_REGISTRY = {
    'xpatch': xPatch,
    'rnn': RNN,
    'lstm': LSTM,
    'gru': GRU,
    'mlp': MLP,
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
