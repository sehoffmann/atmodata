import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='atmodata',
    version='0.0.1',
    author='Sebastian Hoffmann',
    description='A collection of high performant data loading tools for spatio-temporal ML based on torchdata',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Development Status :: 1 - Planning',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.9',
    py_modules=["atmodata"],
    #package_dir={'':''},
    install_requires=[]
)