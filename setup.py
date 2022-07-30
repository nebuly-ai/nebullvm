from pathlib import Path
from setuptools import setup, find_packages


REQUIREMENTS = [
    "numpy>=1.20.0, <1.23.0",
    "scipy<=1.5.4",
    "onnx>=1.10.0",
    "onnxmltools>=1.11.0",
    "py-cpuinfo>=8.0.0",
    "PyYAML>=6.0",
    "psutil>=5.9.0",
    "requests>=2.28.1",
    "tensorflow>=2.7.0",
    "tf2onnx>=1.8.4",
    "torch>=1.10.0",
    "tqdm>=4.63.0",
]


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="nebullvm",
    version="0.4.0",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
)
