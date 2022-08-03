from tempfile import TemporaryDirectory

import pytest
import torch

from nebullvm.base import DeepLearningFramework
from nebullvm.inference_learners.tensor_rt import (
    PytorchTensorRTInferenceLearner,
)
from nebullvm.inference_learners.tensor_rt import NVIDIA_INFERENCE_LEARNERS
from nebullvm.optimizers import TensorRTOptimizer
from nebullvm.optimizers.tests.utils import get_onnx_model, get_torch_model


@pytest.mark.parametrize(
    ("output_library", "dynamic"),
    [
        (DeepLearningFramework.PYTORCH, True),
        (DeepLearningFramework.PYTORCH, False),
    ],
)
def test_tensor_rt(output_library: DeepLearningFramework, dynamic: bool):
    if not torch.cuda.is_available():
        # no need of testing the tensor rt optimizer on devices not
        # supporting CUDA.
        return
    with TemporaryDirectory() as tmp_dir:
        model_path, model_params = get_onnx_model(tmp_dir, dynamic)
        optimizer = TensorRTOptimizer()
        model = optimizer.optimize(model_path, output_library, model_params)
        assert isinstance(model, NVIDIA_INFERENCE_LEARNERS[output_library])

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = NVIDIA_INFERENCE_LEARNERS[output_library].load(tmp_dir)
        assert isinstance(
            loaded_model, NVIDIA_INFERENCE_LEARNERS[output_library]
        )

        inputs_example = list(model.get_inputs_example())
        res = model.predict(*inputs_example)
        assert res is not None

        if dynamic:
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model.predict(*inputs_example)
            assert res is not None


@pytest.mark.parametrize(
    ("dynamic",),
    [
        (True,),
        (False,),
    ],
)
def test_torch_tensor_rt(dynamic: bool):
    if not torch.cuda.is_available():
        # no need of testing the tensor rt optimizer on devices not
        # supporting CUDA.
        return
    with TemporaryDirectory() as tmp_dir:
        model_path, model_params = get_torch_model(dynamic)
        optimizer = TensorRTOptimizer()
        model = optimizer.optimize_from_torch(model_path, model_params)
        assert isinstance(model, PytorchTensorRTInferenceLearner)

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = PytorchTensorRTInferenceLearner.load(tmp_dir)
        assert isinstance(loaded_model, PytorchTensorRTInferenceLearner)

        inputs_example = list(model.get_inputs_example())
        res = model.predict(*inputs_example)
        assert res is not None

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model.predict(*inputs_example)
            assert res is not None
