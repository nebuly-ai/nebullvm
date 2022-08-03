import copy
import logging
from abc import ABC, abstractmethod
from logging import Logger
from typing import Dict, List, Any, Callable, Tuple, Optional

import cpuinfo
import numpy as np
import tensorflow as tf
import torch.nn
from tqdm import tqdm

from nebullvm.base import (
    DeepLearningFramework,
    ModelParams,
    QuantizationType,
    ModelCompiler,
    OptimizationTime,
)
from nebullvm.compressors.base import BaseCompressor
from nebullvm.compressors.intel import TorchIntelPruningCompressor
from nebullvm.compressors.sparseml import SparseMLCompressor
from nebullvm.inference_learners import (
    BaseInferenceLearner,
    PytorchBaseInferenceLearner,
)
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import (
    BaseOptimizer,
    ApacheTVMOptimizer,
    COMPILER_TO_OPTIMIZER_MAP,
    DeepSparseOptimizer,
)
from nebullvm.optimizers.blade_disc import BladeDISCOptimizer
from nebullvm.optimizers.pytorch import PytorchBackendOptimizer
from nebullvm.optimizers.tensor_rt import TensorRTOptimizer
from nebullvm.optimizers.tensorflow import TensorflowBackendOptimizer
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.compilers import (
    tvm_is_available,
    select_compilers_from_hardware_onnx,
    deepsparse_is_available,
    bladedisc_is_available,
    torch_tensorrt_is_available,
)
from nebullvm.utils.data import DataManager
from nebullvm.utils.feedback_collector import FEEDBACK_COLLECTOR


class Step(ABC):
    """Fundamental building block for the Pipeline.

    Attributes:
        logger (Logger, optional): Logger defined by the user.
    """

    def __init__(self, logger: Logger = None):
        self._logger = logger

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict:
        """Run the pipeline step."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    def _log_info(self, text: str):
        if self._logger is None:
            logging.info(text)
        else:
            self._logger.info(text)

    def _log_warning(self, text: str):
        if self._logger is None:
            logging.warning(text)
        else:
            self._logger.warning(text)


class CompressorStep(Step, ABC):
    """Object managing the Compressor step in the Pipeline. This step manages
    all the defined Compressor objects available considering the data given
    by the user.

    Attributes:
        config_file (str, optional): The configuration file containing the
            configuration parameter for each Compressor. The config_file is
            a YAML file having as main keywords the Compressor names and as
            values dictionaries containing the specific parameters for the
            related Compressor object.
        logger (Logger, optional): Logger defined by the user.
    """

    def __init__(self, config_file: str = None, logger: Logger = None):
        super().__init__(logger)
        self._config_file = config_file

    def run(
        self,
        model: Any = None,
        input_data: DataManager = None,
        metric_drop_ths: float = None,
        metric: Callable = None,
        **kwargs,
    ) -> Dict:
        """Run the CompressorStep.

        Args:
            model (Any): Model to be compressed.
            input_data (DataManager): Data to be used for compressing the
                model.
            metric_drop_ths: Maximum reduction in the selected metric accepted.
                No model with an higher error will be accepted. Note that the
                maximum error is modified and then propagated to the next
                steps.
            metric (Callable): Metric to be used for estimating the error
                due to the compression.
            kwargs (Dict): Keyword arguments propagated to the next step.
        """
        compressor_dict = self._get_compressors()
        self._log_info(f"Compressions: {tuple(compressor_dict.keys())}")
        models = {"no_compression": (copy.deepcopy(model), metric_drop_ths)}
        train_input_data, eval_input_data = input_data.split(0.8)
        for technique, compressor in tqdm(compressor_dict.items()):
            try:
                compressed_model, ths = compressor.compress(
                    model,
                    train_input_data,
                    eval_input_data,
                    metric_drop_ths,
                    metric,
                )
                models[technique] = (compressed_model, ths)
            except Exception as ex:
                self._log_warning(
                    f"Error during compression {technique}. Got error {ex}. "
                    f"The compression technique will be skipped. "
                    f"Please consult the documentation for further info or "
                    f"open an issue on GitHub for receiving assistance."
                )
        return {
            "models": models,
            "input_data": eval_input_data,
            "metric": metric,
            **kwargs,
        }

    @abstractmethod
    def _get_compressors(self) -> Dict[str, BaseCompressor]:
        raise NotImplementedError()

    @property
    def name(self):
        return "compression_step"


class TorchCompressorStep(CompressorStep):
    """Object managing the Compressor step in the Pipeline for PyTorch models.
    This step manages all the defined Compressor objects available considering
    the data given by the user.

    At the current state this step supports pruning with SparseML and (just on
    intel devices) pruning with the IntelNeuralCompressor.

    Attributes:
        config_file (str, optional): The configuration file containing the
            configuration parameter for each Compressor. The config_file is
            a YAML file having as main keywords the Compressor names and as
            values dictionaries containing the specific parameters for the
            related Compressor object.
        logger (Logger, optional): Logger defined by the user.
    """

    def _get_compressors(self) -> Dict[str, BaseCompressor]:
        compressors = {
            "sparseml": SparseMLCompressor(config_file=self._config_file)
        }
        # TODO: Reactivate the intel-neural-compressor when properly tested
        if False and "intel" in cpuinfo.get_cpu_info()["brand_raw"].lower():
            compressors["intel_pruning"] = TorchIntelPruningCompressor(
                config_file=self._config_file
            )
        return compressors


class NoCompressionStep(Step):
    """Step to be used when no compression is required.

    Attributes:
        logger (Logger, optional): Logger defined by the user.
    """

    def run(
        self, model: Any, metric_drop_ths: Optional[float], **kwargs
    ) -> Dict:
        return {
            "models": {"no_compression": (model, metric_drop_ths)},
            **kwargs,
        }

    @property
    def name(self):
        return "no_compression"


class OptimizerStep(Step, ABC):
    """Object managing the Optimizers in the pipeline step. All available
    optimizers are run on the model given as input and a list of tuples
    (optimized_model, latency) is given as output.

    Attributes:
        logger (Logger, optional): Logger defined by the user.
    """

    def run(
        self,
        models: Dict[str, Tuple[Any, Optional[float]]] = None,
        output_library: DeepLearningFramework = None,
        model_params: ModelParams = None,
        input_tfms: MultiStageTransformation = None,
        metric: Optional[Callable] = None,
        input_data: Optional[DataManager] = None,
        ignore_compilers: List[ModelCompiler] = None,
        optimization_time: OptimizationTime = None,
        **kwargs,
    ) -> Dict:
        """Run the OptimizerStep for all the available compilers.

        Args:
            models (Dict): Dictionary of models produced by the CompressorStep.
                For each model produced by the previous step, the updated
                metric_drop_ths (i.e. the error allowed on the model) is
                given together with the model. Keys represent the compression
                technique used for obtaining the model.
            output_library (DeepLearningFramework): The target framework.
            model_params (ModelParams): The model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            metric (Callable): Metric to be used for estimating the error
                due to the compression.
            input_data (DataManager): Input data to be used for optimizing the
                model.
            ignore_compilers (List): List of compilers to be ignored.
            optimization_time (OptimizationTime): The optimization time mode.
                It can be either 'constrained' or 'unconstrained'. For
                'unconstrained' optimization all the compilers are re-used on
                the different framework interfaces, even if the model has
                already been compiled with the same compiler on another
                framework interface.
            kwargs (Dict): Extra keywords that will be ignored.
        """

        optimizers = self._get_optimizers(ignore_compilers)
        self._log_info(
            f"Optimizations: "
            f"{tuple(compiler.value for compiler in optimizers.keys())}"
        )
        optimized_models = []

        for prev_tech, (model, metric_drop_ths) in tqdm(models.items()):
            self._log_info(f"Optimizing output of {prev_tech}")
            if model is None:
                continue
            if metric_drop_ths is not None:
                q_types = [
                    None,
                    QuantizationType.DYNAMIC,
                    QuantizationType.HALF,
                ]
                if input_data is not None:
                    q_types.append(QuantizationType.STATIC)
            else:
                q_types = [None]
            for compiler, optimizer in tqdm(optimizers.items()):
                for q_type in q_types:
                    try:
                        optimized_model = self._run_optimizer(
                            optimizer,
                            model,
                            output_library,
                            model_params,
                            input_tfms.copy(),
                            metric_drop_ths,
                            q_type,
                            metric,
                            input_data,
                        )
                        if optimized_model is not None:
                            latency = compute_optimized_running_time(
                                optimized_model
                            )
                        else:
                            latency = np.inf
                        optimized_models.append((optimized_model, latency))
                        if (
                            compiler not in ignore_compilers
                            and optimization_time
                            is OptimizationTime.CONSTRAINED
                        ):
                            ignore_compilers.append(compiler)
                        FEEDBACK_COLLECTOR.store_compiler_result(
                            compiler=compiler,
                            q_type=q_type,
                            metric_drop_ths=metric_drop_ths,
                            latency=latency,
                            compression=prev_tech,
                        )
                    except Exception as ex:
                        self._log_warning(
                            f"Compilation failed with {output_library.value} "
                            f"interface of {compiler}. Got error {ex}. "
                            f"If possible the compilation will be re-scheduled"
                            f" with another interface. Please consult the "
                            f"documentation for further info or open an issue "
                            f"on GitHub for receiving assistance."
                        )
                        FEEDBACK_COLLECTOR.store_compiler_result(
                            compiler=compiler,
                            q_type=q_type,
                            metric_drop_ths=metric_drop_ths,
                            latency=None,
                            compression=prev_tech,
                        )

        return {
            "optimized_models": optimized_models,
            "ignore_compilers": ignore_compilers,
        }

    @property
    def name(self):
        return "optimizer_step"

    @abstractmethod
    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
    ) -> BaseInferenceLearner:
        raise NotImplementedError()

    @abstractmethod
    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        raise NotImplementedError()


class TorchOptimizerStep(OptimizerStep):
    """Object managing the Optimizers in the pipeline step supporting PyTorch
    as compiler interface. All available optimizers are run on the model given
    as input and a list of tuples (optimized_model, latency) is given as
    output.

    Attributes:
        logger (Logger, optional): Logger defined by the user.
    """

    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        optimizers = {
            ModelCompiler.TORCHSCRIPT: PytorchBackendOptimizer(
                logger=self._logger
            ),
        }
        if (
            tvm_is_available()
            and ModelCompiler.APACHE_TVM not in ignore_compilers
        ):
            optimizers[ModelCompiler.APACHE_TVM] = ApacheTVMOptimizer(
                logger=self._logger
            )
        if (
            deepsparse_is_available()
            and ModelCompiler.DEEPSPARSE not in ignore_compilers
        ):
            optimizers[ModelCompiler.DEEPSPARSE] = DeepSparseOptimizer()
        if (
            bladedisc_is_available()
            and ModelCompiler.BLADEDISC not in ignore_compilers
        ):
            optimizers[ModelCompiler.BLADEDISC] = BladeDISCOptimizer()
        if (
            torch_tensorrt_is_available()
            and ModelCompiler.TENSOR_RT not in ignore_compilers
        ):
            optimizers[ModelCompiler.TENSOR_RT] = TensorRTOptimizer()
        return optimizers

    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
    ) -> PytorchBaseInferenceLearner:
        if hasattr(optimizer, "optimize_from_torch"):
            optimized_model = optimizer.optimize_from_torch(
                torch_model=model,
                model_params=model_params,
                metric_drop_ths=metric_drop_ths
                if quantization_type is not None
                else None,
                metric=metric,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
            )
        else:
            optimized_model = optimizer.optimize(
                model=model,
                output_library=output_library,
                model_params=model_params,
                metric_drop_ths=metric_drop_ths
                if quantization_type is not None
                else None,
                metric=metric,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
            )
        return optimized_model


class TFOptimizerStep(OptimizerStep):
    """Object managing the Optimizers in the pipeline step supporting
    TensorFlow as compiler interface. All available optimizers are run on
    the model given as input and a list of tuples (optimized_model, latency)
    is given as output.

    Attributes:
        logger (Logger, optional): Logger defined by the user.
    """

    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        optimizers = {
            ModelCompiler.TFLITE: TensorflowBackendOptimizer(
                logger=self._logger
            )
        }
        return optimizers

    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
    ) -> PytorchBaseInferenceLearner:
        if hasattr(optimizer, "optimize_from_tf"):
            optimized_model = optimizer.optimize_from_tf(
                torch_model=model,
                model_params=model_params,
                metric_drop_ths=metric_drop_ths
                if quantization_type is not None
                else None,
                metric=metric,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
            )
        else:
            optimized_model = optimizer.optimize(
                model=model,
                output_library=output_library,
                model_params=model_params,
                metric_drop_ths=metric_drop_ths
                if quantization_type is not None
                else None,
                metric=metric,
                quantization_type=quantization_type,
                input_tfms=input_tfms,
                input_data=input_data,
            )
        return optimized_model


class OnnxOptimizerStep(OptimizerStep):
    """Object managing the Optimizers in the pipeline step supporting ONNX
    as compiler interface. All available optimizers are run on the model given
    as input and a list of tuples (optimized_model, latency) is given as
    output.

    Attributes:
        logger (Logger, optional): Logger defined by the user.
    """

    def _get_optimizers(
        self, ignore_compilers: List[ModelCompiler]
    ) -> Dict[ModelCompiler, BaseOptimizer]:
        compilers = select_compilers_from_hardware_onnx()
        optimizers = {
            compiler: COMPILER_TO_OPTIMIZER_MAP[compiler](self._logger)
            for compiler in compilers
            if compiler not in ignore_compilers
        }
        return optimizers

    def _run_optimizer(
        self,
        optimizer,
        model: Any,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
    ) -> PytorchBaseInferenceLearner:
        optimized_model = optimizer.optimize(
            model=model,
            output_library=output_library,
            model_params=model_params,
            metric_drop_ths=metric_drop_ths
            if quantization_type is not None
            else None,
            metric=metric,
            quantization_type=quantization_type,
            input_tfms=input_tfms,
            input_data=input_data,
        )
        return optimized_model


class Pipeline(Step):
    """Pipeline object.

    A Pipeline is a list of steps executed sequentially, where each step
    takes as input the output of the previous one.

    Attributes:
        pipeline_name: str,
        steps (List): List of Steps composing the pipeline.
        logger (Logger): Logger defined by the user.
    """

    def __init__(
        self, pipeline_name: str, steps: List[Step], logger: Logger = None
    ):
        super().__init__(logger)
        self._name = pipeline_name
        self._steps = steps

    def run(self, **kwargs) -> Dict:
        self._log_info(f"Running pipeline: {self.name}")
        for step in self._steps:
            self._log_info(f"Running step: {step.name}")
            kwargs = step.run(**kwargs)
        return kwargs

    @property
    def name(self):
        return self._name


def _get_compressor_step(
    model: Any,
    optimization_time: OptimizationTime,
    config_file: Optional[str],
    metric_drop_ths: Optional[float],
    metric: Optional[Callable],
    logger: Optional[Logger],
) -> Step:
    if optimization_time is OptimizationTime.CONSTRAINED:
        return NoCompressionStep(logger=logger)
    if metric_drop_ths is None or metric is None:
        return NoCompressionStep(logger=logger)
    elif isinstance(model, torch.nn.Module):
        return TorchCompressorStep(config_file=config_file, logger=logger)
    else:  # default is NoCompression
        return NoCompressionStep(logger=logger)


def _get_optimizer_step(
    model: Any,
    logger: Optional[Logger],
) -> Step:
    if isinstance(model, torch.nn.Module):
        return TorchOptimizerStep(logger=logger)
    elif isinstance(model, tf.Module):
        return TFOptimizerStep(logger=logger)
    else:
        return OnnxOptimizerStep(logger=logger)


def _get_pipeline_name(model: Any):
    if isinstance(model, torch.nn.Module):
        return "pytorch_pipeline"
    elif isinstance(model, tf.Module):
        return "tensorflow_pipeline"
    else:
        return "onnx_pipeline"


def build_pipeline_from_model(
    model: Any,
    optimization_time: OptimizationTime,
    metric_drop_ths: Optional[float],
    metric: Optional[Callable],
    config_file: Optional[str],
    logger: Logger = None,
) -> Pipeline:
    """Function for building a pipeline from a model and user-defined
    parameters

    Args:
        model (Any): The input model.
        optimization_time (OptimizationTime): The optimization time mode.
            It can be either 'constrained' or 'unconstrained'. For
            'constrained' mode just compilers and precision reduction
            techniques are used (no compression). 'Unconstrained' optimization
            allows the usage of more time consuming techniques as pruning and
            distillation.
        metric_drop_ths (float, optional): Maximum reduction in the
            selected metric accepted. No model with an higher error will be
            accepted, i.e. all optimized model having a larger error respect to
            the original one will be discarded, without even considering their
            possible speed-up.
        metric (Callable): Metric to be used for estimating the error
            due to the optimization techniques.
        config_file (str, optional): Configuration file containing the
            parameters needed for defining the CompressionStep in the pipeline.
        logger (Logger, optional): Logger defined by the user.
    """
    compressor_step = _get_compressor_step(
        model, optimization_time, config_file, metric_drop_ths, metric, logger
    )
    optimizer_step = _get_optimizer_step(model, logger)
    pipeline = Pipeline(
        pipeline_name=_get_pipeline_name(model),
        logger=logger,
        steps=[compressor_step, optimizer_step],
    )
    return pipeline
