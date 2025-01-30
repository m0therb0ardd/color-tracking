from setuptools import find_packages
from setuptools import setup

setup(
    name='nuturtlebot_msgs',
    version='0.0.2',
    packages=find_packages(
        include=('nuturtlebot_msgs', 'nuturtlebot_msgs.*')),
)
