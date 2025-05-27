from setuptools import setup
from setuptools import find_packages

setup(
    name='oasis',
    version='1.0.0',
    description='Optimal Sensor Arrangements for SLAM',
    author='Pushyami Kaveti',
    author_email='p.kaveti@northeastern.edu',
    url='',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'oasis = optimal_camera_placement.main_expectation:main'
        ],

    },
)