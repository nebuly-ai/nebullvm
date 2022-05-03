from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch

from nebullvm.base import DeepLearningFramework, ModelParams, QuantizationType
from nebullvm.config import NVIDIA_FILENAMES, NO_COMPILER_INSTALLATION
from nebullvm.inference_learners.tensor_rt import (
    NVIDIA_INFERENCE_LEARNERS,
    NvidiaInferenceLearner,
)
from nebullvm.optimizers.base import (
    BaseOptimizer,
)
from nebullvm.optimizers.quantization.onnx import quantize_onnx
from nebullvm.optimizers.quantization.tensor_rt import TensorRTCalibrator
from nebullvm.optimizers.quantization.utils import (
    check_precision,
    check_quantization,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.onnx import (
    get_input_names,
    get_output_names,
    create_model_inputs_onnx,
    run_onnx_model,
    convert_to_numpy,
)

if torch.cuda.is_available():
    try:
        import tensorrt as trt
    except ImportError:
        from nebullvm.installers.installers import install_tensor_rt
        import warnings

        if not NO_COMPILER_INSTALLATION:
            warnings.warn(
                "No TensorRT valid installation has been found. "
                "Trying to install it from source."
            )
            install_tensor_rt()
            import tensorrt as trt
        else:
            warnings.warn(
                "No TensorRT valid installation has been found. "
                "It won't be possible to use it in the following."
            )


class TensorRTOptimizer(BaseOptimizer):
    """Class for compiling the AI models on Nvidia GPUs using TensorRT."""

    def _build_and_save_the_engine(
        self,
        engine_path: str,
        onnx_model_path: str,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation,
        quantization_type: QuantizationType = None,
        input_data: List[Tuple[np.ndarray, ...]] = None,
    ):
        # -- Build phase --
        nvidia_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(nvidia_logger)
        # create network definition
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        # build the engine
        # TODO: setup config value for the class in a config file
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if quantization_type is QuantizationType.HALF:
            config.set_flag(trt.BuilderFlag.FP16)
        elif quantization_type is QuantizationType.STATIC:
            assert input_data is not None, (
                "You need to specify the calibration data for "
                "performing static quantization."
            )
            calibrator = TensorRTCalibrator(
                batch_size=model_params.batch_size,
                input_data=input_data,
            )
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
        elif quantization_type is QuantizationType.DYNAMIC:
            onnx_model_path, _ = quantize_onnx(
                onnx_model_path, quantization_type, input_tfms, input_data
            )
            config.set_flag(trt.BuilderFlag.kINT8)
        # import the model
        parser = trt.OnnxParser(network, nvidia_logger)
        success = parser.parse_from_file(onnx_model_path)

        if not success:
            for idx in range(parser.num_errors):
                if self.logger is not None:
                    self.logger.debug(parser.get_error(idx))
            raise ValueError(
                f"Errors occurred while processing the "
                f"ONNX file at {onnx_model_path}"
            )

        if model_params.dynamic_info is not None:
            profile = builder.create_optimization_profile()
            for input_name, input_dynamic_info, input_info in zip(
                get_input_names(onnx_model_path),
                model_params.dynamic_info.inputs,
                model_params.input_infos,
            ):
                profile.set_shape(
                    input_name,
                    (
                        min(model_params.batch_size, 1)
                        if 0 in input_dynamic_info
                        else model_params.batch_size,
                        *(
                            shape
                            if i + 1 not in input_dynamic_info
                            else (input_info.min_sizes or {}).get(i + 1, 1)
                            for i, shape in enumerate(input_info.size)
                        ),
                    ),
                    (model_params.batch_size, *input_info.size),
                    (model_params.batch_size, *input_info.size),
                )
            config.add_optimization_profile(profile)
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

    def optimize(
        self,
        onnx_model: str,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        quantization_ths: float = None,
        quantization_type: QuantizationType = None,
    ) -> Optional[NvidiaInferenceLearner]:
        """Optimize the input model with TensorRT.

        Args:
            onnx_model (str): Path to the saved onnx model.
            output_library (str): DL Framework the optimized model will be
                compatible with.
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            quantization_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used.

        Returns:
            TensorRTInferenceLearner: Model optimized with TensorRT. The model
                will have an interface in the DL library specified in
                `output_library`.
        """
        if not torch.cuda.is_available():
            raise SystemError(
                "You are trying to run an optimizer developed for NVidia gpus "
                "on a machine not connected to any GPU supporting CUDA."
            )
        check_quantization(quantization_type, quantization_ths)
        engine_path = Path(onnx_model).parent / NVIDIA_FILENAMES["engine"]
        self._build_and_save_the_engine(
            engine_path=engine_path,
            onnx_model_path=onnx_model,
            model_params=model_params,
            input_tfms=input_tfms,
            quantization_type=quantization_type,
            input_data=[
                tuple(
                    create_model_inputs_onnx(
                        model_params.batch_size, model_params.input_infos
                    )
                )
            ]
            if quantization_ths is not None
            and quantization_type is QuantizationType.STATIC
            else None,
        )
        model = NVIDIA_INFERENCE_LEARNERS[output_library].from_engine_path(
            input_tfms=input_tfms,
            network_parameters=model_params,
            engine_path=engine_path,
            input_names=get_input_names(onnx_model),
            output_names=get_output_names(onnx_model),
        )
        if quantization_type is not None:
            input_data = [model.get_inputs_example()]
            output_data = [
                tuple(
                    run_onnx_model(
                        onnx_model,
                        [convert_to_numpy(x) for x in input_data[0]],
                    )
                )
            ]
            is_valid = check_precision(
                model, input_data, output_data, quantization_ths
            )
            if not is_valid:
                return None
        return model
