from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow==2.4.1',
    'tensorflow-datasets==4.2.0',
    'scikit-learn>=0.20.2',
    'pandas==0.24.2',
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(include=['trainer', 'trainer.*', 'trainer.**']),
    include_package_data=True,
    description='AI Platform | Training | scikit-learn | Base'
)