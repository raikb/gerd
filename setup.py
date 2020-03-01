import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='gerd',
    version='0.1.0',
    author='Raik Becker',
    description='Easy-to-use multi-area power market model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/raikb/gerd',
    include_package_data=True,
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
