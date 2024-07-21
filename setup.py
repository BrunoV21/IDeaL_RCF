import setuptools
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "gdown==5.2.0",
    "matplotlib==3.7.1",
    "polars==1.0.0",
    "scikit-learn==1.2.2",
    "tqdm==4.41.1",
    "typing_extensions==4.8.0"
]

# Add platform-specific dependencies
if sys.platform.startswith('win'):
    install_requires.extend([        
        "protobuf==3.19.6",
        "tensorflow==2.10.1",
        "tensorflow-addons==0.18.0",
        "tensorflow-estimator==2.10.0",
        "tensorflow-io-gcs-filesystem==0.31.0",
        "numpy==1.24.3",
    ])

elif sys.platform.startswith('linux'):
    install_requires.extend([
        "numpy==1.25.2",
        "tensorflow==2.15.0"
    ])

setuptools.setup(
    name="ideal_rcf",
    version="0.3a",
    author="bruno_v",
    author_email="bruno.vitorino@tecnico.ulisboa.pt",
    description="A python framework for invariant rans turbulence closure.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BrunoV21/IDeaL_RCF",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=(
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)