from tempfile import TemporaryDirectory

import pytest

from nebullvm.base import DeepLearningFramework
from nebullvm.inference_learners.blade_disc import BladeDISCInferenceLearner
from nebullvm.optimizers import BladeDISCOptimizer
from nebullvm.optimizers.tests.utils import get_torch_model


@pytest.mark.parametrize(
    ("output_library", "dynamic"),
    [
        (DeepLearningFramework.PYTORCH, True),
        (DeepLearningFramework.PYTORCH, False),
    ],
)
def test_bladedisc(output_library: DeepLearningFramework, dynamic: bool):
    with TemporaryDirectory() as tmp_dir:
        model_path, model_params = get_torch_model(dynamic)
        optimizer = BladeDISCOptimizer()
        model = optimizer.optimize(model_path, output_library, model_params)
        assert isinstance(model, BladeDISCInferenceLearner)

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = BladeDISCInferenceLearner.load(tmp_dir)
        assert isinstance(loaded_model, BladeDISCInferenceLearner)

        inputs_example = model.get_inputs_example()
        res = model.predict(*inputs_example)
        assert res is not None

        if dynamic:
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model.predict(*inputs_example)
            assert res is not None
