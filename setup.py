"""
TheCatBouncer - AI-powered pet access control system

Setup configuration for packaging and distribution.
"""
from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="thecatbouncer",
    version="2.0.0",
    author="JojiAce",
    author_email="joji@example.com",  # Replace with actual email
    description="AI-powered pet access control system that recognizes your own cat by color & shape and automatically repels intruders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JojiAce/TheCatBouncer",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'thecatbouncer=src.main:main',
        ],
    },
    keywords='ai, computer-vision, pet-control, yolo, object-detection',
    project_urls={
        'Bug Reports': 'https://github.com/JojiAce/TheCatBouncer/issues',
        'Source': 'https://github.com/JojiAce/TheCatBouncer',
    },
)