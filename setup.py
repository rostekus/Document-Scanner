from setuptools import find_packages
from setuptools import setup

setup(
    name='Document Scanner',
    version='1.0.0',
    description='ocr document scannner',
    author='Rostyslav Mosorov',
    author_email='rmosorov@icloud.com',
    license='MIT License',
    url='https://github.com/rostekus/Document-Scanner',
    packages=find_packages(where='scanner'),
    package_dir={'': 'scanner'}

)
