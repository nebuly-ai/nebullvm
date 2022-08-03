import os
from typing import Tuple

import torch

from nebullvm.api.functions import _extract_info_from_data
from nebullvm.api.huggingface import convert_hf_model
from nebullvm.base import ModelParams, DeepLearningFramework
from nebullvm.converters.torch_converters import convert_torch_to_onnx
from nebullvm.utils.data import DataManager
from transformers import AlbertModel, AlbertTokenizer

INPUT_SHAPE = (3, 256, 256)
OUTPUT_SHAPE = (2,)
STATIC_BATCH_SIZE = 1
DYNAMIC_BATCH_SIZE = 2


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3
        )
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3
        )
        self.relu2 = torch.nn.ReLU()
        self.fcn = torch.nn.Linear(32, 2)

    def forward(self, input_tensor_0, input_tensor_1):
        x0 = self.relu2(self.conv2(self.relu1(self.conv1(input_tensor_0))))
        x1 = self.relu2(self.conv2(self.relu1(self.conv1(input_tensor_1))))
        x = x0 + x1
        x = self.fcn(x.mean(dim=(-2, -1)).view(-1, 32))
        return x


def _build_static_model() -> Tuple[torch.nn.Module, ModelParams]:
    model_params = {
        "batch_size": STATIC_BATCH_SIZE,
        "input_infos": [
            {"size": INPUT_SHAPE, "dtype": "float"},
            {"size": INPUT_SHAPE, "dtype": "float"},
        ],
        "output_sizes": [OUTPUT_SHAPE],
    }
    model_params = ModelParams(**model_params)
    model = TestModel()
    return model, model_params


def _build_dynamic_model() -> Tuple[torch.nn.Module, ModelParams]:
    model = TestModel()
    model_params = {
        "batch_size": DYNAMIC_BATCH_SIZE,
        "input_infos": [
            {"size": INPUT_SHAPE, "dtype": "float"},
            {"size": INPUT_SHAPE, "dtype": "float"},
        ],
        "output_sizes": [OUTPUT_SHAPE],
        "dynamic_info": {
            "inputs": [{0: "batch_size"}, {0: "batch_size"}],
            "outputs": [{0: "batch_size"}],
        },
    }
    return model, ModelParams(**model_params)


def get_onnx_model(temp_dir: str, dynamic: bool = False):
    model_path = os.path.join(temp_dir, "test_model.onnx")
    if dynamic:
        model, model_params = _build_dynamic_model()
    else:
        model, model_params = _build_static_model()
    convert_torch_to_onnx(model, model_params, model_path)
    return model_path, model_params


def get_torch_model(dynamic: bool = False):
    if dynamic:
        model, model_params = _build_dynamic_model()
    else:
        model, model_params = _build_static_model()
    return model, model_params


def get_huggingface_model(temp_dir: str, dl_framework: DeepLearningFramework):
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    model = AlbertModel.from_pretrained("albert-base-v1")

    text = "Short text you wish to process"
    encoded_input = tokenizer(text, return_tensors="pt")

    (
        model,
        input_data,
        input_names,
        output_structure,
        output_type,
    ) = convert_hf_model(model, [encoded_input])

    input_data = DataManager(input_data)

    model_path = os.path.join(temp_dir, "test_model.onnx")

    model_params = _extract_info_from_data(
        model,
        input_data,
        dl_framework,
        None,
    )

    convert_torch_to_onnx(model, model_params, model_path, input_data)

    return (
        model_path,
        model_params,
        output_structure,
        input_names,
        output_type,
        input_data,
    )
