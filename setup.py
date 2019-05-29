from setuptools import setup
with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()
setup(name='pathflowai',
      version='0.1',
      description='A modular approach for preprocessing and deep learning on histopathology images.',
      url='https://github.com/jlevy44/PathFlowAI',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=[],
      entry_points={
            'console_scripts':['']
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['pathflowai'],
      install_requires=[])
