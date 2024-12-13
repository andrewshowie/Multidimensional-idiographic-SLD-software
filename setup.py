# setup.py
from setuptools import setup, find_packages

setup(
    name="linguistic-analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.11.0',
        'spacy>=3.0.0',
        'pandas>=1.3.0',
        'numpy>=1.19.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'nltk>=3.6.0',
        'networkx>=2.6.0',
        'scipy>=1.7.0',
        'tkinter',  # Usually comes with Python
    ],
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="A linguistic analysis suite for real-time text processing",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Topic :: Text Processing :: Linguistic",
    ],
)

