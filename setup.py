from setuptools import setup, find_packages

setup(
   name='iaif',
   version='1.0',
   author='Markus Klar',
   author_email='markus.klar@glasgow.ac.uk',
   packages=['iaif'],
   package_data={'': ['**']},
   url='https://github.com/mkl4r/iaif',
   license='LICENSE',
   python_requires='>=3.11',
   install_requires=[
       "numpy", "optax", "tqdm", "pyyaml", "matplotlib", "pandas", "difai @ git+https://github.com/mkl4r/difai-base.git@main"
   ],
   extras_require={
    "gpu": ["jax[cuda13]"],
    "tpu": ["jax[tpu]"],
    "cpu": ["jax"],
    },
)