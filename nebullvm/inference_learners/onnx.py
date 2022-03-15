import os
import shutil
import warnings
from abc import ABC
from pathlib import Path
from typing import Union, List, Generator, Tuple, Dict, Type

import cpuinfo
import numpy as np
import tensorflow as tf
import torch

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.config import ONNX_FILENAMES
from nebullvm.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
)

try:
    import onnxruntime as ort
except ImportError:
    warnings.warn(
        "No valid onnxruntime installation found. Trying to install it..."
    )
    from nebullvm.installers.installers import install_onnxruntime

    install_onnxruntime()
    import onnxruntime as ort


def _is_intel_cpu():
    if torch.cuda.is_available():
        return False  # running on GPU
    cpu_info = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" in cpu_info:
        return True
    return False


class ONNXInferenceLearner(BaseInferenceLearner, ABC):
    """Model converted to ONNX and run with Microsoft's onnxruntime.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        onnx_path (str or Path): Path to the onnx model.
        input_names (List[str]): Input names used when the onnx model
            was produced.
        output_names (List[str]): Output names used when the onnx model
            was produced.
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        input_names: List[str],
        output_names: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.onnx_path = onnx_path
        if _is_intel_cpu():
            ort_session_options = ort.SessionOptions()
            ort_session_options.add_session_config_entry(
                "session.set_denormal_as_zero", "1"
            )
            ort_session = ort.InferenceSession(onnx_path, ort_session_options)
        else:
            ort_session = ort.InferenceSession(onnx_path)
        self._session = ort_session
        self.input_names = input_names
        self.output_names = output_names

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path or str): Path to the directory where the model will
                be stored.
            kwargs (Dict): Dictionary of key-value pairs that will be saved in
                the model metadata file.
        """
        metadata = LearnerMetadata.from_model(
            self,
            input_names=self.input_names,
            output_names=self.output_names,
            **kwargs,
        )
        metadata.save(path)

        shutil.copy(
            self.onnx_path,
            os.path.join(str(path), ONNX_FILENAMES["model_name"]),
        )

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        """Load the model.

        Args:
            path (Path or str): Path to the directory where the model is
                stored.
            kwargs (Dict): Dictionary of additional arguments for consistency
                with other Learners.

        Returns:
            ONNXInferenceLearner: The optimized model.
        """
        if len(kwargs) > 0:
            warnings.warn(
                f"No extra keywords expected for the load method. "
                f"Got {kwargs}."
            )
        onnx_path = os.path.join(str(path), ONNX_FILENAMES["model_name"])
        metadata = LearnerMetadata.read(path)

        return cls(
            network_parameters=ModelParams(**metadata.network_parameters),
            onnx_path=onnx_path,
            input_names=metadata["input_names"],
            output_names=metadata["output_names"],
        )

    def _predict_arrays(self, input_arrays: Generator[np.ndarray, None, None]):
        input_dict = {
            name: input_array
            for name, input_array in zip(self.input_names, input_arrays)
        }
        outputs = self._session.run(self.output_names, input_dict)
        return outputs


class PytorchONNXInferenceLearner(
    ONNXInferenceLearner, PytorchBaseInferenceLearner
):
    """Model run with Microsoft's onnxruntime using a Pytorch interface.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        onnx_path (str or Path): Path to the onnx model.
        input_names (List[str]): Input names used when the onnx model
            was produced.
        output_names (List[str]): Output names used when the onnx model
            was produced.
    """

    def predict(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[Tensor]): Input tensors belonging to the same
                batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[Tensor]: Output tensors. Note that the output tensors does
                not correspond to the prediction on the input tensors with a
                1 to 1 mapping. In fact the output tensors are produced as the
                multiple-output of the model given a (multi-) tensor input.
        """
        input_arrays = (
            input_tensor.cpu().detach().numpy()
            for input_tensor in input_tensors
        )
        outputs = self._predict_arrays(input_arrays)
        return tuple(torch.from_numpy(output) for output in outputs)


class TensorflowONNXInferenceLearner(
    ONNXInferenceLearner, TensorflowBaseInferenceLearner
):
    """Model run with Microsoft's onnxruntime using a tensorflow interface.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        onnx_path (str or Path): Path to the onnx model.
        input_names (List[str]): Input names used when the onnx model
            was produced.
        output_names (List[str]): Output names used when the onnx model
            was produced.
    """

    def predict(self, *input_tensors: tf.Tensor) -> Tuple[tf.Tensor]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[Tensor]): Input tensors belonging to the same
                batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[Tensor]: Output tensors. Note that the output tensors does
                not correspond to the prediction on the input tensors with a
                1 to 1 mapping. In fact the output tensors are produced as the
                multiple-output of the model given a (multi-) tensor input.
        """
        input_arrays = (input_tensor.numpy() for input_tensor in input_tensors)
        outputs = self._predict_arrays(input_arrays)
        # noinspection PyTypeChecker
        return tuple(tf.convert_to_tensor(output) for output in outputs)


ONNX_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[ONNXInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchONNXInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowONNXInferenceLearner,
}
