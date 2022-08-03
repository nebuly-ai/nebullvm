from tempfile import TemporaryDirectory

import pytest

from nebullvm.base import DeepLearningFramework
from nebullvm.inference_learners.pytorch import PytorchBackendInferenceLearner
from nebullvm.optimizers.pytorch import PytorchBackendOptimizer
from nebullvm.optimizers.tests.utils import get_torch_model


@pytest.mark.parametrize(
    ("output_library", "dynamic"),
    [
        (DeepLearningFramework.PYTORCH, True),
        (DeepLearningFramework.PYTORCH, False),
    ],
)
def test_torchscript(output_library: DeepLearningFramework, dynamic: bool):
    with TemporaryDirectory() as tmp_dir:
        model_path, model_params = get_torch_model(dynamic)
        optimizer = PytorchBackendOptimizer()
        model = optimizer.optimize(model_path, output_library, model_params)
        assert isinstance(model, PytorchBackendInferenceLearner)

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = PytorchBackendInferenceLearner.load(tmp_dir)
        assert isinstance(loaded_model, PytorchBackendInferenceLearner)

        inputs_example = list(model.get_inputs_example())
        res = model.predict(*inputs_example)
        assert res is not None

        if dynamic:  # Check also with a smaller bath_size
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model.predict(*inputs_example)
            assert res is not None
