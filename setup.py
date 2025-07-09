#!/usr/bin/env python3
"""
Setup script for Vietnamese SLU Data Augmentation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vietnamese-slu-augmentation",
    version="1.0.0",
    author="Nguyen Tran Gia Ky",
    author_email="ntgiaky@gmail.com",
    description="A comprehensive data augmentation system for Vietnamese SLU tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/vietnamese-slu-augmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Vietnamese",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.html", "*.css", "*.js"],
    },
    entry_points={
        "console_scripts": [
            "vn-slu-augment=scripts.1_generate_augmentations:main",
            "vn-slu-integrate=scripts.3_integrate_approved_data:main",
        ],
    },
    keywords=[
        "vietnamese",
        "slu",
        "spoken language understanding",
        "data augmentation",
        "nlp",
        "machine learning",
        "smart home",
        "voice assistant",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/vietnamese-slu-augmentation/issues",
        "Source": "https://github.com/your-username/vietnamese-slu-augmentation",
        "Documentation": "https://github.com/your-username/vietnamese-slu-augmentation/docs",
    },
)