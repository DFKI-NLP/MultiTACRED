from io import open

from setuptools import find_packages, setup

setup(
    name="multi-tacred",
    version="0.0.1",
    author="Leonhard Hennig, Gabriel Kressin, Phuc Tran Truong",
    author_email="firstname.lastname@dfki.de",
    description="MultiTACRED - A Multilingual Version of the TAC Relation Extraction Dataset",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="relation extraction, information extraction, machine translation",
    license="MIT",
    url="https://github.com/DFKI-NLP/MultiTACRED",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "hydra-core==1.2.0",
        "google-cloud-translate==2.0.1",
        "python-dotenv>=0.20.0",
        "spacy==3.1.6",
        "trankit==1.1.1",
        "torch==1.10.2",  # install with pip install . --extra-index-url https://download.pytorch.org/whl/cu113
        "sherlock @ git+https://git@github.com/DFKI-NLP/sherlock.git@0.2.1#egg=sherlock",
        "pre-commit",  # hooks for applying linters on commit
        "black==22.6.0",  # to fix dependency conflicts
        "typer==0.4.2",  # to fix dependency conflicts
    ],
    tests_require=["flake8"],
    entry_points={"console_scripts": []},
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
