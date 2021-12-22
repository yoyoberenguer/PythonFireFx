"""
Setup.py file

Configure the project, build the package and upload the package to PYPI
"""
import sys
import setuptools
from Cython.Build import cythonize
from setuptools import Extension

# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
                      "\nTry: \n   C:\\pip install numpy on a window command prompt.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


extra_compile_args = []

if sys.platform == 'win32':
    extra_compile_args = ["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"]

elif sys.platform == 'darwin':
    pass

elif sys.platform == 'linux2':
    pass

elif sys.platform == 'cygwin':
    pass


setuptools.setup(
    name="PythonFireFx",
    version="1.0.1",
    author="Yoann Berenguer",
    author_email="yoyoberenguer@hotmail.com",
    description="Python procedural fire effect",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyoberenguer/PythonFireFx",
    # packages=setuptools.find_packages(),
    packages=['PythonFireFx'],
    ext_modules=cythonize([
        Extension("PythonFireFx.FireFx", ["PythonFireFx/FireFx.pyx"],
                  extra_compile_args=extra_compile_args,
                  language="c")

    ]),

    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    license='GNU General Public License v3.0',

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Cython',

        # Pick your license as you wish
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        # 'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],

    install_requires=[
        'setuptools>=49.2.1',
        'Cython>=0.28',
        'numpy>=1.18',
        'pygame>=2.0'
    ],
    python_requires='>=3.6',
    platforms=['Windows'],
    include_package_data=True,
    data_files=[
        ('./lib/site-packages/PythonFireFx',
         ['LICENSE',
          'MANIFEST.in',
          'pyproject.toml',
          'README.md',
          'requirements.txt',
          'PythonFireFx/__init__.py',
          'PythonFireFx/__init__.pxd',
          'PythonFireFx/setup_FireFx.py',
          'PythonFireFx/FireFx.pyx',
          'PythonFireFx/FireFx.pxd'

          ]),
        ('./lib/site-packages/PythonFireFx/Include',
         ['PythonFireFx/Include/Shaderlib.c'
          ]),

        ('./lib/site-packages/PythonFireFx/Assets',
         [
            'PythonFireFx/Assets/background.jpg'
         ])
    ],

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/yoyoberenguer/PythonFireFx/issues',
        'Source': 'https://github.com/yoyoberenguer/PythonFireFx',
    },
)

