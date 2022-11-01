from setuptools import setup

setup(
    name='Feature Selection With Reinforcement Learning',
    version='1.0.0',    
    description='Here is an implementation of a feature selection algorithm based on the reinforcement learning method.',
    url='https://github.com/blefo/FS_RL',
    author='Baptiste Lefort',
    author_email='baptiste.lefort@icloud.com',
    license='BSD 2-clause',
    packages=['feature_selection_RL'],
    install_requires=['matplotlib',
                      'numpy',
                      'sklearn'                     
                      ],
)