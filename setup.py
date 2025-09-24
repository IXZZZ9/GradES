"""Setup script for GradES package."""

from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="grades",
    version="1.0.2",
    description="Gradient-based Early Stopping for Efficient Fine-tuning of Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IXZZZ9/GradES",
    project_urls={
        "Bug Tracker": "https://github.com/IXZZZ9/GradES/issues",
        "Documentation": "https://github.com/IXZZZ9/GradES#readme",
        "Source Code": "https://github.com/IXZZZ9/GradES",
        "Paper": "https://arxiv.org/abs/2509.01842",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "wandb": ["wandb>=0.12.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    keywords=[
        "machine learning",
        "deep learning",
        "transformers",
        "early stopping",
        "gradient-based",
        "fine-tuning",
        "llm",
        "pytorch",
        "huggingface",
        "efficiency",
    ],
    include_package_data=True,
    zip_safe=False,
)