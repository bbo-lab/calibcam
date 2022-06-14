import pathlib
from setuptools import setup
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="bbo-calibcam",
    version="2.1.0",
    description="Calibrate intrinsic and extrinsic parameters of cameras with charuco boards",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bbo-lab/calibcam",
    author="BBO-lab @ caesar",
    author_email="kay-michael.voit@caesar.de",
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=['calibcam'],
    include_package_data=True,
    install_requires=["numpy", "pyyaml", "scipy", "bbo-calibcamlib", "jax", "jaxlib", "imageio", "bbo-ccvtools"],
)
