from typing import Dict, Type

from nebullvm.base import ModelCompiler
from nebullvm.optimizers.base import BaseOptimizer  # noqa F401
from nebullvm.optimizers.blade_disc import BladeDISCOptimizer  # noqa F401
from nebullvm.optimizers.deepsparse import DeepSparseOptimizer  # noqa F401
from nebullvm.optimizers.onnx import ONNXOptimizer  # noqa F401
from nebullvm.optimizers.openvino import OpenVinoOptimizer  # noqa F401
from nebullvm.optimizers.tensor_rt import TensorRTOptimizer  # noqa F401
from nebullvm.optimizers.tvm import ApacheTVMOptimizer  # noqa F401

__all__ = [k for k in globals().keys() if not k.startswith("_")]


COMPILER_TO_OPTIMIZER_MAP: Dict[ModelCompiler, Type[BaseOptimizer]] = {
    ModelCompiler.APACHE_TVM: ApacheTVMOptimizer,
    ModelCompiler.OPENVINO: OpenVinoOptimizer,
    ModelCompiler.TENSOR_RT: TensorRTOptimizer,
    ModelCompiler.ONNX_RUNTIME: ONNXOptimizer,
    ModelCompiler.DEEPSPARSE: DeepSparseOptimizer,
    ModelCompiler.BLADEDISC: BladeDISCOptimizer,
}
