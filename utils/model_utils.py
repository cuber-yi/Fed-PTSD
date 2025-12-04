from src.model.xPatch import xPatch
from src.model.RNN import RNN
from src.model.LSTM import LSTM
from src.model.GRU import GRU
from src.model.MLP import MLP
from src.model.DLinear import DLinear
from src.model.Pyraformer import Pyraformer
from src.model.FEDformer import FEDformer
from src.model.TimesNet import TimesNet
from src.model.PatchTST import PatchTST


MODEL_REGISTRY = {
    'xpatch': xPatch,
    'rnn': RNN,
    'lstm': LSTM,
    'gru': GRU,
    'mlp': MLP,
    'dlinear': DLinear,
    'pyraformer': Pyraformer,
    'fedformer': FEDformer,
    'timesnet': TimesNet,
    'patchtst': PatchTST
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
