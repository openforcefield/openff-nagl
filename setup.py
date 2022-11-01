"""
OpenFF NAGL
A short description of the project.
"""
import sys
from setuptools import setup, find_namespace_packages
import versioneer

short_description = "A short description of the project.".strip().split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name='openff-nagl',
    author='Lily Wang',
    author_email='lily.wang@openforcefield.org',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_namespace_packages(include=['openff.*']),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    python_requires=">=3.8",          # Python version restrictions
    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,
    # Required packages, pulls from pip if needed
    # do not use for Conda deployment
    install_requires=[
        "click",
        "click-option-group",
        "tqdm",
        "openff-toolkit",
        "openff-units",
        "pydantic",
        "rdkit",
    ],
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='openff-nagl.readthedocs.io/en/latest/',  # Website
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

    extras_require={
        "test": [
            "pytest>=6.0",
            "pytest-xdist>=2.5",
            "pytest-cov>=3.0",
        ],
        "doc": [
            "sphinx>=1.8",
            "openff-sphinx-theme @ git+https://github.com/openforcefield/openff-sphinx-theme.git@main",
        ]
    },

    entry_points={
        "console_scripts": [
            "openff-nagl=openff.nagl.cli:cli",
        ]
    }
)
