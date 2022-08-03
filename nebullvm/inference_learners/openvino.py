import json
import shutil
import warnings
from abc import ABC
from pathlib import Path
from typing import Dict, Union, Type, Generator, Tuple, List

import cpuinfo
import numpy as np
import tensorflow as tf
import torch

from nebullvm.base import ModelParams, DeepLearningFramework
from nebullvm.config import (
    OPENVINO_FILENAMES,
    NO_COMPILER_INSTALLATION,
)
from nebullvm.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
    TensorflowBaseInferenceLearner,
    NumpyBaseInferenceLearner,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager

try:
    from openvino.runtime import Core, Model, CompiledModel, InferRequest
except ImportError:
    if (
        "intel" in cpuinfo.get_cpu_info()["brand_raw"].lower()
        and not NO_COMPILER_INSTALLATION
    ):
        warnings.warn(
            "No valid OpenVino installation has been found. "
            "Trying to re-install it from source."
        )
        from nebullvm.installers.installers import install_openvino

        install_openvino(with_optimization=True)
        from openvino.runtime import Core, Model, CompiledModel, InferRequest
    else:
        warnings.warn(
            "No Openvino library detected. "
            "The Openvino Inference learner should not be used."
        )
        Model = CompiledModel = InferRequest = object


class OpenVinoInferenceLearner(BaseInferenceLearner, ABC):
    """Model optimized using ApacheTVM.

    The class cannot be directly instantiated, but implements all the core
    methods needed for using ApacheTVM at inference time.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
    """

    def __init__(
        self,
        model: Model,
        compiled_model: CompiledModel,
        infer_request: InferRequest,
        input_keys: List,
        output_keys: List,
        description_file: str,
        weights_file: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.compiled_model = compiled_model
        self.infer_request = infer_request
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.description_file = self._store_file(description_file)
        self.weights_file = self._store_file(weights_file)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        """Load the model.

        Args:
            path (Path or str): Path to the directory where the model is
                stored.
            kwargs (Dict): Dictionary of additional arguments for the
                `from_model_name` class method.

        Returns:
            OpenVinoInferenceLearner: The optimized model.
        """
        path = Path(path)
        with open(path / OPENVINO_FILENAMES["metadata"], "r") as fin:
            metadata = json.load(fin)
        metadata.update(kwargs)
        metadata["network_parameters"] = ModelParams(
            **metadata["network_parameters"]
        )
        input_tfms = metadata.get("input_tfms")
        if input_tfms is not None:
            metadata["input_tfms"] = MultiStageTransformation.from_dict(
                input_tfms
            )
        model_name = str(path / OPENVINO_FILENAMES["description_file"])
        model_weights = str(path / OPENVINO_FILENAMES["weights"])
        return cls.from_model_name(
            model_name=model_name, model_weights=model_weights, **metadata
        )

    @classmethod
    def from_model_name(
        cls,
        network_parameters: ModelParams,
        model_name: str,
        model_weights: str,
        input_tfms: MultiStageTransformation = None,
        input_data: DataManager = None,
        **kwargs,
    ):
        """Build the optimized model from the network description and its
        weights.

        Args:
            network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
            model_name (str): File containing a description of the optimized
                model.
            model_weights (str): File containing the model weights.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            input_data (DataManager, optional): User defined data.
        """
        if len(kwargs) > 0:
            warnings.warn(f"Found extra parameters: {kwargs}")

        core = Core()
        model = core.read_model(model=model_name, weights=model_weights)
        compiled_model = core.compile_model(model=model, device_name="CPU")
        infer_request = compiled_model.create_infer_request()

        input_keys = list(
            map(lambda obj: obj.get_any_name(), compiled_model.inputs)
        )
        output_keys = list(
            map(lambda obj: obj.get_any_name(), compiled_model.outputs)
        )

        return cls(
            model,
            compiled_model,
            infer_request,
            input_keys,
            output_keys,
            input_tfms=input_tfms,
            network_parameters=network_parameters,
            description_file=model_name,
            weights_file=model_weights,
            input_data=input_data,
        )

    def _rebuild_network(self, input_shapes: Dict):
        # get the input name and input shape info from model
        model_inputs_info = {
            item.get_any_name(): tuple(item.shape)
            for item in self.model.inputs
        }
        if all(
            input_shape == model_inputs_info[input_name]
            for input_name, input_shape in input_shapes.items()
        ):
            # If the new input shapes is equal to the previous one do nothing.
            return

        core = Core()
        model = core.read_model(
            model=self.description_file, weights=self.weights_file
        )
        model.reshape(input_shapes)
        compiled_model = core.compile_model(model=model, device_name="CPU")
        infer_request = compiled_model.create_infer_request()

        self.model = model
        self.compiled_model = compiled_model
        self.infer_request = infer_request

    def _get_metadata(self, **kwargs) -> LearnerMetadata:
        # metadata = {
        #     key: self.__dict__[key] for key in ("input_keys", "output_keys")
        # }
        metadata = {}
        metadata.update(kwargs)
        return LearnerMetadata.from_model(self, **metadata)

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path or str): Path to the directory where the model will
                be stored.
            kwargs (Dict): Dictionary of key-value pairs that will be saved in
                the model metadata file.
        """
        path = Path(path)
        path.mkdir(exist_ok=True)
        metadata = self._get_metadata(**kwargs)
        with open(path / OPENVINO_FILENAMES["metadata"], "w") as fout:
            json.dump(metadata.to_dict(), fout)

        shutil.copy(
            self.description_file,
            path / OPENVINO_FILENAMES["description_file"],
        )
        shutil.copy(self.weights_file, path / OPENVINO_FILENAMES["weights"])

    def _predict_array(
        self,
        input_arrays: Generator[np.ndarray, None, None],
        input_shapes: Generator[Tuple[int, ...], None, None] = None,
    ) -> Generator[np.ndarray, None, None]:
        if input_shapes is not None:
            input_shapes_dict = {
                name: size for name, size in zip(self.input_keys, input_shapes)
            }
            self._rebuild_network(input_shapes_dict)

        results = self.infer_request.infer(
            inputs={
                input_key: input_array
                for input_key, input_array in zip(
                    self.input_keys, input_arrays
                )
            }
        )
        results = {
            output_key.get_any_name(): output_arr
            for output_key, output_arr in results.items()
        }

        return (results[output_key] for output_key in self.output_keys)


class PytorchOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, PytorchBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a Pytorch interface.

    This class can be used exactly in the same way as a pytorch Module object.
    At prediction time it takes as input pytorch tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
    """

    def run(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
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
        extra_kwargs = {}
        if self.network_parameters.dynamic_info is not None:
            extra_kwargs["input_shapes"] = (
                tuple(input_tensor.size()) for input_tensor in input_tensors
            )
        output_arrays = self._predict_array(input_arrays, **extra_kwargs)
        return tuple(
            torch.from_numpy(output_array) for output_array in output_arrays
        )


class TensorflowOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, TensorflowBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a tensorflow interface.

    This class can be used exactly in the same way as a tf.Module or
    keras.Model object.
    At prediction time it takes as input tensorflow tensors given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
    """

    def run(self, *input_tensors: tf.Tensor) -> Tuple[tf.Tensor, ...]:
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
        extra_kwargs = {}
        if self.network_parameters.dynamic_info is not None:
            extra_kwargs["input_shapes"] = (
                tuple(input_tensor.shape) for input_tensor in input_tensors
            )
        output_arrays = self._predict_array(input_arrays, **extra_kwargs)
        # noinspection PyTypeChecker
        return tuple(
            tf.convert_to_tensor(output_array)
            for output_array in output_arrays
        )


class NumpyOpenVinoInferenceLearner(
    OpenVinoInferenceLearner, NumpyBaseInferenceLearner
):
    """Model optimized using ApacheTVM with a numpy interface.

    This class can be used exactly in the same way as a sklearn or
    numpy-based model.
    At prediction time it takes as input numpy arrays given as positional
    arguments.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        exec_network (any): The graph executor. This is the
            central component in the OpenVino optimized model execution.
        input_keys (List): Keys associated to the inputs.
        output_keys (List): Keys associated to the outputs.
        description_file (str): File containing a description of the optimized
            model.
        weights_file (str): File containing the model weights.
    """

    def run(self, *input_tensors: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[np.ndarray]): Input tensors belonging to
                the same batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[np.ndarray]: Output tensors. Note that the output tensors
                does not correspond to the prediction on the input tensors
                with a 1 to 1 mapping. In fact the output tensors are produced
                as the multiple-output of the model given a (multi-) tensor
                input.
        """
        input_arrays = (input_tensor for input_tensor in input_tensors)
        extra_kwargs = {}
        if self.network_parameters.dynamic_info is not None:
            extra_kwargs["input_shapes"] = (
                tuple(input_tensor.shape) for input_tensor in input_tensors
            )
        output_arrays = self._predict_array(input_arrays, **extra_kwargs)
        return tuple(output_arrays)


OPENVINO_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[OpenVinoInferenceLearner]
] = {
    DeepLearningFramework.PYTORCH: PytorchOpenVinoInferenceLearner,
    DeepLearningFramework.TENSORFLOW: TensorflowOpenVinoInferenceLearner,
    DeepLearningFramework.NUMPY: NumpyOpenVinoInferenceLearner,
}
