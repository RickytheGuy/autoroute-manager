from setuptools import setup, find_packages

setup(
    name='autoroute_manager',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "gradio",
        "xarray",
        "matplotlib",
        "geopandas",
        "gdal>=3.8",
        "pyyaml",
        "tqdm",
    ],
)
