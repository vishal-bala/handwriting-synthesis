from setuptools import setup, find_packages

setup(
    name='handwriting-synthesis',
    version='0.1.5',
    packages=find_packages(),
    python_requires='>=3.0, <4.0',
    package_data={
        '': ['*.npy'],
    },
    include_package_data=True,
    install_requires=[
        'CairoSVG',
        'pandas',
        'Pillow',
        'scikit-learn',
        'scipy',
        'svgwrite',
        'tensorflow==2.3.1',
        'tensorflow-probability==0.6.0',
    ],
)
