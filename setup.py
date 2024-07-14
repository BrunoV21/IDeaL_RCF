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
    classifiers=(
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)