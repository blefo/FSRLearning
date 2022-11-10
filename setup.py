from setuptools import setup

setup(
    name='FSRLearning',
    version='1.0.5',    
    description='Here is an implementation of a feature selection algorithm based on the reinforcement learning method.',
    url='***',
    author='Baptiste Lefort',
    author_email='***',
    license='MIT',
    readme = 'README.md',
    packages=['FSRLearning'],
    install_requires=['matplotlib',
                      'numpy',
                      'scikit-learn',
                      'tqdm'                     
                      ],
)
