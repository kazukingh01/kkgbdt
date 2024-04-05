from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkgbdt*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkgbdt',
    version='1.2.6',
    description='my GBDT (Gradient Boosting Decision Tree) liblary. Supported LightGBM and XGBoost.',
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
        'pandas==2.2.1',
        'numpy==1.26.4',
        "xgboost==2.0.3",
        "lightgbm==4.3.0",
        "optuna==3.6.0",
        'setuptools>=62.0.0',
        'wheel>=0.37.0',
    ],
    python_requires='>=3.12.2'
)
