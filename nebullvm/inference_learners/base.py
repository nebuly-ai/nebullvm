import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, InitVar
from pathlib import Path
from tempfile import mkdtemp
from typing import Union, Dict, Any, List, Optional

import numpy as np
import tensorflow as tf
import torch

from nebullvm.base import ModelParams
from nebullvm.config import LEARNER_METADATA_FILENAME
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.onnx import create_model_inputs_onnx
from nebullvm.utils.tf import create_model_inputs_tf
from nebullvm.utils.torch import create_model_inputs_torch


@dataclass
class BaseInferenceLearner(ABC):
    """Base class for Inference Learners."""

    network_parameters: ModelParams
    input_tfms: Optional[MultiStageTransformation] = None
    input_data: InitVar[List[Any]] = None

    def __post_init__(self, input_data):
        if self.input_tfms is not None and len(self.input_tfms) < 0:
            self.input_tfms = None
        self._tmp_folder = Path(mkdtemp())
        self._input_data = input_data

    def _store_file(self, file_path: Union[str, Path]):
        return shutil.copy(str(file_path), str(self._tmp_folder))

    def _store_dir(self, dir_path: Union[str, Path]):
        try:
            # For python >= 3.8
            return shutil.copytree(
                str(dir_path), str(self._tmp_folder), dirs_exist_ok=True
            )
        except TypeError:
            # For python <=3.7
            if os.path.isdir(self._tmp_folder):
                shutil.rmtree(str(self._tmp_folder))
            return shutil.copytree(str(dir_path), str(self._tmp_folder))

    def __del__(self):
        shutil.rmtree(self._tmp_folder, ignore_errors=True)

    def predict_from_files(
        self, input_files: List[str], output_files: List[str]
    ):
        """Get a model prediction from file.

        The input file is read, processed and a prediction is run on top of it.
        The prediction is then returned into another file (in the same
        directory of the input file itself).

        Args:
            input_files (List[str]): List of paths to the input file.
            output_files (List[str]): List of paths to the file storing
                the prediction.
        """
        inputs = (self._read_file(input_file) for input_file in input_files)
        preds = self(*inputs)
        for pred, output_file in zip(preds, output_files):
            self._save_file(pred, output_file)

    def predict_from_listified_tensors(self, *listified_tensors: List):
        """Predict from listified tensor.

        Method useful to be used in services receiving the input tensor
        from an HTTP call.

        Args:
            listified_tensors (List): List of list-like version of the
                input tensors. Note that each element of the external list is
                a listified input tensor.

        Returns:
            List: List of list-like predictions.
        """
        inputs = (
            self.list2tensor(listified_tensor)
            for listified_tensor in listified_tensors
        )
        if self.input_tfms is not None:
            inputs = (self.input_tfms(_input) for _input in inputs)
        preds = self.predict(*inputs)
        return [self.tensor2list(pred) for pred in preds]

    def list2tensor(self, listified_tensor: List) -> Any:
        """Convert list to tensor.

        Args:
            listified_tensor (List): Listified version of the input tensor.

        Returns:
            Any: Tensor for the prediction.
        """
        raise NotImplementedError()

    def tensor2list(self, tensor: Any) -> List:
        """Convert tensor to list.

        Args:
            tensor (any): Input tensor.

        Returns:
            List: Listified version of the tensor.
        """
        raise NotImplementedError()

    def _read_file(self, input_file: str) -> Any:
        """Read tensor from file.
        Args:
            input_file (str): Path to the file containing the input tensor.

        Returns:
            Any: Tensor read from the file.
        """
        raise NotImplementedError()

    def _save_file(self, prediction: Any, output_file: str):
        """Save prediction in the appropriate format.

        Args:
            prediction (any): The predicted tensor.
            output_file (str): Path to the file where storing the prediction.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Any:
        """Take as input a tensor and returns a prediction"""
        return self(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Abstract method implementing the prediction code."""
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        """Alternative method to the predict one."""
        return self(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.input_tfms is not None:
            args = (self.input_tfms(_input) for _input in args)
        return self.run(*args, **kwargs)

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path): Path to the directory where saving the model.
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        """Load the model.

        Args:
            path (Path): Path to the directory where the model is stored.

        Returns:
            BaseInferenceLearner: Loaded model.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_inputs_example(self):
        """The function returns an example of the input for the optimized
        model predict method.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_format(self):
        return ".txt"

    @property
    @abstractmethod
    def input_format(self):
        return ".txt"


class LearnerMetadata:
    """Class for storing all the metadata about a model.

    The stored information can be used for loading the appropriate model.

    Attributes:
        class_name (str): Name of the model class. For instance, for the model
            object `CustomModel()`, the class name is 'CustomModel'.
        module_name (str): Path to the python module where the model class
            is defined.
        network_parameters (Dict): Dictionaty containing the network
            parameters, i.e. batch_size, input_size and output_size.
        kwargs: External attributes that will be stored in the Metadata file.
    """

    NAME: str = LEARNER_METADATA_FILENAME
    class_name: str
    module_name: str

    def __init__(
        self,
        class_name: str,
        module_name: str,
        network_parameters: Union[ModelParams, Dict],
        input_tfms: Union[MultiStageTransformation, Dict] = None,
        **kwargs,
    ):
        self.class_name = class_name
        self.module_name = module_name
        self.network_parameters = (
            network_parameters.dict()
            if isinstance(network_parameters, ModelParams)
            else network_parameters
        )
        self.input_tfms = (
            input_tfms.to_dict()
            if isinstance(input_tfms, MultiStageTransformation)
            else input_tfms
        )
        self.__dict__.update(**kwargs)

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError(
                f"Error in key type. Expected str got {type(item)}"
            )
        elif item.startswith("_"):
            raise ValueError("Trying to access a private attribute.")
        return self.__dict__.get(item)

    @classmethod
    def from_model(cls, model: BaseInferenceLearner, **kwargs):
        """Create the metadata from the Inference Learner.

        Args:
            model (BaseInferenceLearner): Model from which extract the
                metadata.
            kwargs: External attributes that will be stored in the Metadata
                file.

        Returns:
            LearnerMetadata: Metadata associated with the model.
        """
        return cls(
            class_name=model.__class__.__name__,
            module_name=model.__module__,
            network_parameters=model.network_parameters,
            input_tfms=model.input_tfms,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Create the metadata file from a dictionary.

        This method is the reverse one of `to_dict`.

        Args:
            dictionary (Dict): Dictionary containing the metadata.

        Returns:
            LearnerMetadata: Metadata associated with the model.
        """
        if any(
            key not in dictionary
            for key in ("class_name", "module_name", "network_parameters")
        ):
            raise ValueError(
                "The input dictionary should contain both the model class "
                "name and module."
            )
        return cls(**dictionary)

    def to_dict(self) -> Dict:
        """Method for converting the LearnerMetadata in a python dictionary.

        Returns:
            Dict: Dictionary containing the metadata.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if (
                len(key) > 0
                and key[0].islower()
                and not key.startswith("_")
                and value is not None
            )
        }

    @classmethod
    def read(cls, path: Union[Path, str]):
        """Read the metadata file and store it into a LearnerMetadata object.

        Args:
            path (Path): Path to the directory containing the metadata file.

        Returns:
            LearnerMetadata: Metadata associated with the model.
        """
        path = Path(path)
        with open(path / cls.NAME, "r") as fin:
            metadata_dict = json.load(fin)
        return cls(**metadata_dict)

    def save(self, path: Union[Path, str]):
        """Save the metadata of the model in a file.

        Args:
            path (Path): Path to the directory where saving the model metadata.
        """
        path = Path(path)
        path.mkdir(exist_ok=True)
        metadata_dict = self.to_dict()
        with open(path / self.NAME, "w") as fout:
            json.dump(metadata_dict, fout)

    def load_model(
        self, path: Union[Path, str], **kwargs
    ) -> BaseInferenceLearner:
        """Method for loading the InferenceLearner from its metadata.

        The ModelMetadata file contains all the information necessary for
        loading the Learner, as it contains both the module where the model
        is defined and the class name of the model object. This method calls
        the appropriate class method of the Model object, thus the actual
        model loading is delegate to its methods.

        Args:
            path (Path): Path to the directory containing the files where
                the model optimization is saved.
            kwargs: Dictionary containing the arguments for the model's load
                function.
        """
        exec(f"from {self.module_name} import {self.class_name}")
        model = eval(self.class_name).load(path=path, **kwargs)
        return model


class PytorchBaseInferenceLearner(BaseInferenceLearner, ABC):
    @property
    def input_format(self):
        return ".pt"

    @property
    def output_format(self):
        return ".pt"

    def list2tensor(self, listified_tensor: List) -> torch.Tensor:
        """Convert list to tensor.

        Args:
            listified_tensor (List): Listified version of the input tensor.

        Returns:
            torch.Tensor: Tensor for the prediction.
        """
        return torch.tensor(listified_tensor)

    def tensor2list(self, tensor: torch.Tensor) -> List:
        """Convert tensor to list.

        Args:
            tensor (any): Input tensor.

        Returns:
            List: Listified version of the tensor.
        """
        return tensor.cpu().detach().numpy().tolist()

    def _read_file(self, input_file: Union[str, Path]) -> torch.Tensor:
        input_tensor = torch.load(input_file)
        return input_tensor

    def _save_file(
        self, prediction: torch.Tensor, output_file: Union[str, Path]
    ):
        torch.save(prediction, output_file)

    def get_inputs_example(self, random=False):
        if self._input_data is None or random:
            return tuple(
                create_model_inputs_torch(
                    batch_size=self.network_parameters.batch_size,
                    input_infos=self.network_parameters.input_infos,
                )
            )
        else:
            return self._input_data


class TensorflowBaseInferenceLearner(BaseInferenceLearner, ABC):
    @property
    def input_format(self):
        return ".npy"

    @property
    def output_format(self):
        return ".npy"

    def list2tensor(self, listified_tensor: List) -> tf.Tensor:
        """Convert list to tensor.

        Args:
            listified_tensor (List): Listified version of the input tensor.

        Returns:
            tf.Tensor: Tensor ready to be used for prediction.
        """
        return tf.convert_to_tensor(listified_tensor)

    def tensor2list(self, tensor: tf.Tensor) -> List:
        """Convert tensor to list.

        Args:
            tensor (tf.Tensor): Input tensor.

        Returns:
            List: Listified version of the tensor.
        """
        return tensor.numpy().tolist()

    def _read_file(self, input_file: Union[str, Path]) -> tf.Tensor:
        numpy_array = np.load(input_file)
        input_tensor = tf.convert_to_tensor(numpy_array)
        return input_tensor

    def _save_file(self, prediction: tf.Tensor, output_file: Union[str, Path]):
        prediction.numpy().save(output_file)

    def get_inputs_example(self, random=False):
        if self._input_data is None or random:
            return tuple(
                create_model_inputs_tf(
                    batch_size=self.network_parameters.batch_size,
                    input_infos=self.network_parameters.input_infos,
                )
            )
        else:
            return self._input_data


class NumpyBaseInferenceLearner(BaseInferenceLearner, ABC):
    @property
    def input_format(self):
        return ".npy"

    @property
    def output_format(self):
        return ".npy"

    def list2tensor(self, listified_tensor: List) -> np.ndarray:
        """Convert list to numpy arrays.

        Args:
            listified_tensor (List): Listified version of the input tensor.

        Returns:
            np.array: Tensor ready to be used for prediction.
        """
        return np.array(listified_tensor)

    def tensor2list(self, tensor: np.ndarray) -> List:
        """Convert tensor to list.

        Args:
            tensor (tf.Tensor): Input tensor.

        Returns:
            List: Listified version of the tensor.
        """
        return tensor.tolist()

    def _read_file(self, input_file: Union[str, Path]) -> np.ndarray:
        numpy_array = np.load(input_file)
        return numpy_array

    def _save_file(
        self, prediction: np.ndarray, output_file: Union[str, Path]
    ):
        np.save(output_file, prediction)

    def get_inputs_example(self, random=False):
        if self._input_data is None or random:
            return tuple(
                create_model_inputs_onnx(
                    batch_size=self.network_parameters.batch_size,
                    input_infos=self.network_parameters.input_infos,
                )
            )
        else:
            return self._input_data


class InferenceLearnerWrapper(BaseInferenceLearner, ABC):
    """Wrapper model around InferenceLearners. It's a base class: cannot be
    instantiated.

    For all the BaseInferenceLearner-related methods, the implementation of
    the core model will be used. This class just re-implement the load and save
    methods, allowing (and forcing) then the child class to re-implement the
    `predict` method.

    Attributes:
        network_parameters (ModelParams): Model parameters.
        core_inference_learner (BaseInferenceLearner): Inference Learner.
    """

    CORE_MODEL_SAVE_DIR = "core_model"

    def __init__(self, core_inference_learner: BaseInferenceLearner):
        super().__init__(
            network_parameters=core_inference_learner.network_parameters
        )
        self.core_inference_learner = core_inference_learner

    def list2tensor(self, listified_tensor: List) -> Any:
        return self.core_inference_learner.list2tensor(listified_tensor)

    def tensor2list(self, tensor: Any) -> List:
        return self.core_inference_learner.tensor2list(tensor)

    def _read_file(self, input_file: str) -> Any:
        return self.core_inference_learner._read_file(input_file)

    def _save_file(self, prediction: Any, output_file: str):
        self.core_inference_learner._save_file(prediction, output_file)

    def save(self, path: Union[str, Path], **kwargs):
        core_model_path = Path(path) / self.CORE_MODEL_SAVE_DIR
        core_model_path.mkdir(exist_ok=True, parents=True)
        self.core_inference_learner.save(core_model_path, **kwargs)
        extra_metadata_kwargs = self._get_extra_metadata_kwargs()
        metadata = LearnerMetadata.from_model(self, **extra_metadata_kwargs)
        metadata.save(path)
        self._save_wrapper_extra_info()

    def _get_extra_metadata_kwargs(self) -> Dict:
        raise NotImplementedError

    def _save_wrapper_extra_info(self):
        raise NotImplementedError

    @staticmethod
    def _convert_metadata_to_inputs(metadata: LearnerMetadata) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _load_wrapper_extra_info(builder_inputs: Dict) -> Dict:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        core_model_path = Path(path) / cls.CORE_MODEL_SAVE_DIR
        core_learner = LearnerMetadata.read(core_model_path).load_model(
            core_model_path, **kwargs
        )
        metadata = LearnerMetadata.read(path)
        input_dict = cls._convert_metadata_to_inputs(metadata)
        input_dict = cls._load_wrapper_extra_info(input_dict)
        input_dict.update({"core_inference_learner": core_learner})
        return cls(**input_dict)

    def get_inputs_example(self):
        return self.core_inference_learner.get_inputs_example()

    @property
    def output_format(self):
        return self.core_inference_learner.output_format

    @property
    def input_format(self):
        return self.core_inference_learner.input_format
