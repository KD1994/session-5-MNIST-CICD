from setuptools import setup, find_packages

# Function to read the requirements.txt file
def parse_requirements(filename):
    """
    Parse the requirements.txt file to get the list of required packages.
    """
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

setup(
    name='mnist_cicd_pipeline',
    version='0.0.1',
    description='A simple CICD pipeline for MNIST digit classification using PyTorch',
    author='Keval Darji',
    author_email='krdarji22@gmail.com',
    url='https://github.com/KD1994/session-5-MNIST-CICD.git',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.6',
)