import sys
import setuptools

# if sys.version_info < (3):
#     sys.exit("Python < 3.0 is not supported")

with open("README.md", 'r') as handle:
    long_description = handle.read()

setuptools.setup(
    name = "OSCR_analysis",
    version = "1.0.0",
    author = "Steven Vanuytsel",
    author_email = "steven_vanuytsel@yahoo.com",
    description = ("Package to analyze oSCR tracks" 
                    "with the possibility of analysing accompanying ephys data."),
    long_description = long_description,
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires='>=3.0',
    install_requires=[
        'numpy>=1.19.1',
        'pandas>=1.1.0',
        'pyabf>=2.2.7',
        'matplotlib>=3.3.1',
        'trackpy>=0.4.2',
        'scikit-image>=0.17.2',
        'scipy>=1.5.0'
    ]
)