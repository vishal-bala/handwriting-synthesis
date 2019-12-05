from setuptools import setup, find_packages

try:  # For pip >= 10
    from pip._internal.download import PipSession
    from pip._internal.req import parse_requirements
except ImportError:  # For pip <= 9.0.3
    from pip.download import PipSession
    from pip.req import parse_requirements

setup(
    name='handwriting-synthesis',
    version='0.1.3',
    packages=find_packages(),
    python_requires='>=3.0, <4.0',
    package_data={
        '': ['*.npy'],
    },
    include_package_data=True,
    install_requires=[
        'CairoSVG',
        'pandas',
        'scikit-learn',
        'scipy',
        'svgwrite',
        'tensorflow==1.13.1',
        'tensorflow-probability==0.6.0',
    ],
)
