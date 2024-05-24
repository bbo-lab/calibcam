import pathlib
from setuptools import setup, find_packages
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

packages=find_packages()
print(packages)
# This call to setup() does all the work
setup(
    name="bbo-calibcam",
    version="3.0.0",
    description="Calibrate intrinsic and extrinsic parameters of cameras with charuco boards",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bbo-lab/calibcam",
    author="BBO-lab @ caesar",
    author_email="kay-michael.voit@mpinb.mpg.de",
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=packages,
    include_package_data=True,
    install_requires=["numpy", "pyyaml", "scipy", "bbo-calibcamlib", "jax", "jaxlib", "imageio", "bbo-ccvtools",
                      "bbo-svidreader", "bbo_bbo"],
)
