from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow==1.15.4',
    'scikit-learn>=0.20.2',
    'pandas==0.24.2',
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(include=['trainer', 'trainer.*', 'trainer.**']),
    include_package_data=True,
    description='AI Platform | Training | scikit-learn | Base',
    # data_files=[('trainer', ['face_mask_recognition_dataset/checksums.tsv'])]
)