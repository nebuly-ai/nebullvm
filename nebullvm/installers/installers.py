import os
import platform
import subprocess
from pathlib import Path

import cpuinfo
import torch


def _get_cpu_arch():
    arch = cpuinfo.get_cpu_info()["arch"].lower()
    if "x86" in arch:
        return "x86"
    else:
        return "arm"


def _get_os():
    return platform.system()


def install_tvm(working_dir: str = None):
    """Helper function for installing ApacheTVM.

    This function needs some prerequisites for running, as a valid `git`
    installation and having MacOS or a Linux-distribution as OS.

    Args:
        working_dir (str, optional): The directory where the tvm repo will be
            cloned and installed.
    """
    path = Path(__file__).parent
    # install pre-requisites
    installation_file_prerequisites = str(
        path / "install_tvm_prerequisites.sh"
    )
    subprocess.run(
        ["bash", installation_file_prerequisites],
        cwd=working_dir or Path.home(),
    )
    installation_file = str(path / "install_tvm.sh")
    hardware_config = _get_cpu_arch()
    if torch.cuda.is_available():
        hardware_config = f"{hardware_config}_cuda"
    env_dict = {
        "CONFIG_PATH": str(
            path / f"tvm_installers/{hardware_config}/config.cmake"
        ),
        **dict(os.environ.copy()),
    }
    subprocess.run(
        ["bash", installation_file],
        cwd=working_dir or Path.home(),
        env=env_dict,
    )


def install_bladedisc():
    """Helper function for installing BladeDisc."""
    has_cuda = False
    if torch.cuda.is_available():
        has_cuda = True

    path = Path(__file__).parent
    installation_file = str(path / "install_bladedisc.sh")
    subprocess.Popen(["bash", installation_file, str(has_cuda).lower()])


def install_torch_tensor_rt():
    """Helper function for installing Torch-TensorRT.

    The function will install the software only if a cuda driver is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Torch-TensorRT can run just on Nvidia machines. "
            "No available cuda driver has been found."
        )

    # Verify that TensorRT is installed, otherwise install it
    try:
        import tensorrt  # noqa F401
    except ImportError:
        install_tensor_rt()

    # Will work when Torch-TensorRT v1.2 will be available
    cmd = [
        "pip3",
        "install",
        "torch-tensorrt",
        "-f",
        "https://github.com/pytorch/TensorRT/releases",
    ]
    subprocess.run(cmd)

    # # Install Torch-TensorRT from alpha wheel, works with python3.7
    # cmd = [
    #     "wget",
    #     "https://output.circle-artifacts.com/output/job/32d63a4c-0c"
    #     "5d-42d0-a150-629ec1f3d376/artifacts/0/x86_64-release-pkgs/"
    #     "torch_tensorrt-1.2.0a0-cp37-cp37m-linux_x86_64.whl",
    # ]
    # subprocess.run(cmd)
    #
    # cmd = [
    #     "pip",
    #     "install",
    #     "./torch_tensorrt-1.2.0a0-cp37-cp37m-linux_x86_64.whl",
    # ]
    # subprocess.run(cmd)
    #
    # os.remove("./torch_tensorrt-1.2.0a0-cp37-cp37m-linux_x86_64.whl")


def install_tensor_rt():
    """Helper function for installing TensorRT.

    The function will install the software only if a cuda driver is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT can run just on Nvidia machines. "
            "No available cuda driver has been found."
        )
    path = Path(__file__).parent
    installation_file = str(path / "install_tensor_rt.sh")
    subprocess.run(["bash", installation_file])


def install_openvino(with_optimization: bool = True):
    """Helper function for installing the OpenVino compiler.

    This function just works on intel machines.

    Args:
        with_optimization (bool): Flag for installing the full openvino engine
            or limiting the installation to the tools need for inference
            models.
    """
    processor = cpuinfo.get_cpu_info()["brand_raw"].lower()
    if "intel" not in processor:
        raise RuntimeError(
            f"Openvino can run just on Intel machines. "
            f"You are trying to install it on {processor}"
        )
    openvino_version = "openvino-dev" if with_optimization else "openvino"
    cmd = ["pip3", "install", f"{openvino_version}[onnx]"]
    subprocess.run(cmd)
    cmd = ["pip3", "install", "numpy>=1.20,<1.23"]
    subprocess.run(cmd)


def install_onnxruntime():
    """Helper function for installing the right version of onnxruntime."""
    distribution_name = "onnxruntime"
    if torch.cuda.is_available():
        distribution_name = f"{distribution_name}-gpu"
    if _get_os() == "Darwin" and _get_cpu_arch() == "arm":
        cmd = ["conda", "install", "-y", distribution_name]
    else:
        cmd = ["pip3", "install", distribution_name]
    subprocess.run(cmd)
    # install requirements for onnxruntime.transformers
    cmd = ["pip3", "install", "coloredlogs", "sympy"]
    subprocess.run(cmd)


def install_deepsparse():
    """Helper function for installing DeepSparse."""
    cmd = ["pip3", "install", "deepsparse"]
    subprocess.run(cmd)
