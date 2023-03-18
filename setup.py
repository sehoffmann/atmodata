import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

packages = setuptools.find_packages(exclude=['test*'])
print('Including packages: ', packages)

setuptools.setup(
    name='atmodata',
    description='A collection of high performant data loading tools for spatio-temporal ML based on torchdata',
    version='0.0.1',
    url='https://github.com/sehoffmann/atmodata',
    author='Sebastian Hoffmann',
    author_email='shoffmann.git@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=packages,
    python_requires='>=3.9',
    py_modules=["atmodata"],
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Development Status :: 1 - Planning',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)