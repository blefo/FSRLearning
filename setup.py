from distutils.core import setup
setup(
  name = 'FSRLearning',
  packages=[
    'FSRLearning',
  ],
  version = '1.0.6',
  license='MIT',
  description = 'The first feature selection method based on reinforcement learning - Python library available on pip for a fast deployment.',
  author = 'Baptiste Lefort',
  author_email = 'lefort.baptiste@icloud.com',
  url = 'https://github.com/blefo/FSRLearning',
  keywords = ['feature', 'selection', 'reinforcement learning', 'large dataset', 'ai'],
  install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
      ],
  classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
  ],
  long_description=open('README.md').read(),
  long_description_content_type='text/markdown',
)