import numpy
import setuptools
from Cython.Build import cythonize
from setuptools.extension import Extension

VER = '0.1.9'
AUTHOR = 'Dylan Skola'

print('*' * 80)
print('* {:<76} *'.format('PEAS {} by {}'.format(VER, AUTHOR)))
print('*' * 80)
print()

with open("README.md", "r") as fh:
    long_description = fh.read()

include_dirs = [numpy.get_include()]

extensions = cythonize([Extension('peas.scoring_funcs_cython',
                        ['peas/scoring_funcs_cython.pyx', 'peas/scoring_funcs_c.c', 'peas/my_array_funcs.c'],
                        extra_compile_args=['-std=gnu99', '-O3', '-march=native', '-finline-functions'],
                        include_dirs=include_dirs)])

#print(extensions[0].include_dirs)
setuptools.setup(name='PEAS',
                 version=VER,
                 description='Proximal Enrichment By Approximated Sampling',
                 long_description=long_description,
                 url='https://github.com/phageghost/PEAS',
                 author=AUTHOR,
                 author_email='peas@phageghost.net',
                 license='MIT',
                 packages=['peas', 'peas.fitapproxdistros'],
                 python_requires='>=3.6',
                 scripts=['scripts/genomic_peas.py'],
                 ext_modules=extensions,
                 include_dirs=[numpy.get_include()],
                 install_requires=['numpy', 'datetime', 'scipy', 'statsmodels', 'pandas', 'matplotlib', 'seaborn',
                                   'empdist'],
                 zip_safe=False,
                 classifiers=(
                             "Programming Language :: Python :: 3",
                             "License :: OSI Approved :: MIT License",
                             "Operating System :: OS Independent",
                             ),
                 )
