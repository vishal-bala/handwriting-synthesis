from setuptools import setup, find_packages

try:  # For pip >= 10
    from pip._internal.download import PipSession
    from pip._internal.req import parse_requirements
except ImportError:  # For pip <= 9.0.3
    from pip.download import PipSession
    from pip.req import parse_requirements

setup(
    name='handwriting-synthesis',
    version='0.1.0',
    packages=find_packages(),
    author='Sean Vasquez',
    author_email='seanjv@mit.edu',
    maintainer='Vishal Bala',
    maintainer_email='vishal@vishalbala.com',
    python_requires='>=3.0, <4.0',
    install_requires=[str(req.req) for req in parse_requirements(
        'requirements.txt', session=PipSession(),
    )],
)
