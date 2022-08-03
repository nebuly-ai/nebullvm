import time
from abc import abstractmethod, ABC
from tqdm import tqdm
from typing import Any, Dict, Type

import numpy as np
import tensorflow as tf
import torch

from nebullvm.api.functions import _check_input_data, _extract_info_from_data
from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import create_model_inputs_onnx
from nebullvm.utils.tf import create_model_inputs_tf
from nebullvm.utils.torch import create_model_inputs_torch


def _get_dl_framework(model: Any):
    if isinstance(model, torch.nn.Module) or str(model).startswith("Pytorch"):
        return DeepLearningFramework.PYTORCH
    elif isinstance(model, tf.Module) or str(model).startswith("Tensorflow"):
        return DeepLearningFramework.TENSORFLOW
    elif isinstance(model, str) or str(model).startswith("Numpy"):
        return DeepLearningFramework.NUMPY
    else:
        raise TypeError(f"Model type {type(model)} not supported.")


def _create_model_inputs(
    dl_framework: DeepLearningFramework, model_params: ModelParams
):
    if dl_framework == DeepLearningFramework.PYTORCH:
        input_data = create_model_inputs_torch(
            model_params.batch_size, model_params.input_infos
        )
    elif dl_framework == DeepLearningFramework.TENSORFLOW:
        input_data = create_model_inputs_tf(
            model_params.batch_size, model_params.input_infos
        )
    elif dl_framework == DeepLearningFramework.NUMPY:
        input_data = create_model_inputs_onnx(
            model_params.batch_size, model_params.input_infos
        )
    else:
        raise TypeError(f"Unknown framework {dl_framework}")

    return input_data


class BaseBenchmark(ABC):
    def __init__(self, model, input_tensors, n_warmup=50, n_runs=1000):
        self.model = model
        self.input_tensors = input_tensors
        self.n_warmup = n_warmup
        self.n_runs = n_runs

    @abstractmethod
    def benchmark(self):
        raise NotImplementedError


class PytorchBenchmark(BaseBenchmark):
    def benchmark(self):
        has_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if has_cuda else "cpu")
        input_tensors = [tensor.to(device) for tensor in self.input_tensors]
        batch_size = input_tensors[0].shape[0]

        with torch.no_grad():
            for _ in tqdm(
                range(self.n_warmup),
                desc=f"Performing warm up on {self.n_warmup} iterations",
            ):
                features = self.model(*input_tensors)
        if has_cuda:
            torch.cuda.synchronize()
        timings = []
        with torch.no_grad():
            for _ in tqdm(
                range(1, self.n_runs + 1),
                desc=f"Performing benchmark on {self.n_runs} iterations",
            ):
                start_time = time.time()
                features = self.model(*input_tensors)
                if has_cuda:
                    torch.cuda.synchronize()
                end_time = time.time()
                timings.append(end_time - start_time)

        if isinstance(features, tuple):
            features = features[0]

        print("Input shapes:", [tensor.shape for tensor in input_tensors])
        print(
            "Output features shapes:", [feature.shape for feature in features]
        )
        print(f"Batch size: {batch_size}")

        throughput = batch_size / np.mean(timings)
        latency = np.mean(timings) / batch_size

        print("Average Throughput: %.2f data/second" % throughput)
        print("Average Latency: %.4f seconds/data" % latency)

        return throughput, latency


class TensorflowBenchmark(BaseBenchmark):
    def benchmark(self):
        # TODO: implement benchmark for tensorflow
        raise NotImplementedError


class NumpyBenchmark(BaseBenchmark):
    def benchmark(self):
        # TODO: implement benchmark for numpy
        raise NotImplementedError


def benchmark(model, input_data, random=False, n_warmup=50, n_runs=1000):
    """Performs a Benchmark on the input model regardless of the framework it
    was used for implementing it.

    Args:
        model (Any): The input model.
        input_data (Iterable or Sequence): Input data to be used for
            optimizing the model. PyTorch, TensorFlow
            and Onnx respectively accept input tensor in `torch.Tensor`,
            `tf.Tensor` and `np.ndarray` formats. Note that the each input
            sample must be a tuple containing a tuple as first element, the
            `inputs`, and the `label` as second element. The `inputs` needs to
            be passed as tuple even if a single input is needed by the model
            (in this case the `inputs` tuple will contain just an element).
            HuggingFace models can take as data samples both dictionaries or
            strings. Strings will then be converted in data samples using the
            HuggingFace tokenizer which must be given as input when just a
            list of string is provided as input_data (tokenizers can be passed
            as extra arguments of this function using the keyword `tokenizer`).
        random (bool, optional): If set to true, the data used to benchmark the
            model will be computed randomly given the info extracted from the
            provided input_data.
        n_warmup (int, optional): Number of warmup iterations.
        n_runs (int, optional): Number of iterations performed to benchmark
            the model.
    """
    dl_framework = _get_dl_framework(model)

    if _check_input_data(input_data):
        input_data = DataManager(input_data)
    else:
        input_data = DataManager.from_iterable(input_data)

    if random:
        model_params = _extract_info_from_data(
            model,
            input_data,
            dl_framework,
            None,
        )
        input_data = _create_model_inputs(dl_framework, model_params)
    else:
        input_data = list(input_data.get_list(1)[0])

    BENCHMARK_FUNCTIONS[dl_framework](
        model=model,
        input_tensors=input_data,
        n_warmup=n_warmup,
        n_runs=n_runs,
    ).benchmark()


BENCHMARK_FUNCTIONS: Dict[DeepLearningFramework, Type[BaseBenchmark]] = {
    DeepLearningFramework.PYTORCH: PytorchBenchmark,
    DeepLearningFramework.TENSORFLOW: TensorflowBenchmark,
    DeepLearningFramework.NUMPY: NumpyBenchmark,
}
