from setuptools import setup, find_packages

VERSION = "0.0.0"

INSTALL_REQUIRES = [
    "numpy",
    "seaborn",
    "matplotlib",
    "pandas"
]

setup(
    name="option_pricers",
    version=VERSION,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
)
