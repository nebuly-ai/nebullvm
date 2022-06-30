from pathlib import Path
from setuptools import setup, find_packages

REQUIREMENTS = [
    "numpy>=1.19.0",
    "onnx>=1.10.0",
    "onnxmltools>=1.11.0",
    "py-cpuinfo>=8.0.0",
    "tensorflow>=2.7.0, <2.8.0",
    "tf2onnx>=1.8.4",
    "torch>=1.10.0",
    "tqdm>=4.63.0",
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nebullvm",
    version="0.3.1",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    package_data={
        # Include all bash files:
        "": ["*.sh"],
        # And include any file needed for config
        "nebullvm": ["*config.cmake"],
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
)
