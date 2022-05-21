from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkgbdt*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkgbdt',
    version='1.0.0',
    description='my object detection library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukingh01/kkgbdt",
    author='kazuking',
    author_email='kazukingh01@gmail.com',
    license='Public License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Private License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
    ],
    python_requires='>=3.8'
)
