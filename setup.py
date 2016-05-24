from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='ebak',
      version='0.1.dev',
      author='adrn',
      author_email='adrn@astro.columbia.edu',
      url='https://github.com/adrn/ebak',
      license="License :: OSI Approved :: MIT License",
      description='Eclipsing binaries.',
      long_description=long_description,
      packages=['ebak'],
)
