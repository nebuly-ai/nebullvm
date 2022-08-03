from tempfile import TemporaryDirectory

import pytest

from nebullvm.base import DeepLearningFramework
from nebullvm.inference_learners.tvm import TVM_INFERENCE_LEARNERS
from nebullvm.optimizers import ApacheTVMOptimizer
from nebullvm.optimizers.tests.utils import get_onnx_model


@pytest.mark.parametrize(
    ("output_library", "dynamic"),
    [
        (DeepLearningFramework.PYTORCH, True),
        (DeepLearningFramework.PYTORCH, False),
    ],
)
def test_tvm(output_library: DeepLearningFramework, dynamic: bool):
    with TemporaryDirectory() as tmp_dir:
        model_path, model_params = get_onnx_model(tmp_dir, dynamic)
        optimizer = ApacheTVMOptimizer()
        model = optimizer.optimize(model_path, output_library, model_params)
        assert isinstance(model, TVM_INFERENCE_LEARNERS[output_library])

        # Test save and load functions
        model.save(tmp_dir)
        loaded_model = TVM_INFERENCE_LEARNERS[output_library].load(tmp_dir)
        assert isinstance(loaded_model, TVM_INFERENCE_LEARNERS[output_library])

        inputs_example = model.get_inputs_example()
        res = model.predict(*inputs_example)
        assert res is not None

        if dynamic:
            inputs_example = [
                input_[: len(input_) // 2] for input_ in inputs_example
            ]
            res = model.predict(*inputs_example)
            assert res is not None
