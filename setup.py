import os
import subprocess
from pathlib import Path

import setuptools

ROOT_DIR = Path(__file__).parent.resolve()


def _get_version():
    with open(ROOT_DIR / 'version.txt') as f:
        version = f.readline().strip()

    sha = "Unknown"
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        pass

    os_build_version = os.getenv("BUILD_VERSION")
    if os_build_version:
        version = os_build_version
    # elif sha != "Unknown":
    #    version += "+" + sha[:7]

    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "atmodata" / "version.py"
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


def _get_requirements():
    req_list = []
    with Path("requirements.txt").open("r") as f:
        for line in f:
            req = line.strip()
            if len(req) == 0 or req.startswith("#"):
                continue
            req_list.append(req)
    return req_list


if __name__ == '__main__':
    version, sha = _get_version()
    _export_version(version, sha)
    print('Building version: ', version)

    requirements = _get_requirements()

    packages = setuptools.find_packages(exclude=['test*'])
    print('Including packages: ', packages)

    with open(ROOT_DIR / 'README.md') as f:
        long_description = f.read()

    setuptools.setup(
        name='atmodata',
        description='A collection of high performant data loading tools for spatio-temporal ML based on torchdata',
        version=version,
        url='https://github.com/sehoffmann/atmodata',
        author='Sebastian Hoffmann',
        author_email='shoffmann.git@gmail.com',
        long_description=long_description,
        long_description_content_type='text/markdown',
        python_requires='>=3.8',
        install_requires=requirements,
        packages=packages,
        entry_points={
            'console_scripts': [
                'atmodata-precompute=atmodata.cli:cli_precompute',
            ],
        },
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: Other/Proprietary License',
            'Operating System :: OS Independent',
            'Development Status :: 3 - Alpha',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
        ],
    )
