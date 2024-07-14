import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ideal_rcf",
    version="0.1a",
    author="bruno_v",
    author_email="bruno.vitorino@tecnico.ulisboa.pt",
    description="A python framework for invariant rans turbulence closure.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/demopackage",
    packages=setuptools.find_packages(),
    install_requires=[
        "gdown==5.2.0",
        "matplotlib==3.7.1",
        "numpy==1.24.3",
        "polars==1.0.0",
        "protobuf==3.19.6",
        "scikit-learn==1.2.2",
        "tensorflow==2.10.1",
        "tensorflow-addons==0.18.0",
        "tensorflow-estimator==2.10.0",
        "tensorflow-io-gcs-filesystem==0.31.0",
        "tqdm==4.41.1",
        "typing_extensions==4.8.0"
    ],
    classifiers=(
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)