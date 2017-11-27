from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = [r.rstrip('\n') for r in f.readlines()]


setup(
    name='bankruptcy',
    version='0.0.1',
    description='Bankruptcy prediction',
    url='https://www.asidatascience.com',
    author='ASI Data Science',
    author_email='engineering@asidatascience.com',
    packages=find_packages(),
    install_requires=requirements
)

