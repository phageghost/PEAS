import setuptools

VER = '0.0.1'
AUTHOR = 'Dylan Skola'

print('*' * 80)
print('* {:<76} *'.format('PEAS {} by {}'.format(VER, AUTHOR)))
print('*' * 80)
print()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='PEAS',
                 version=VER,
                 description=' Proximal Enrichment By Approximated Sampling  ',
                 long_description=long_description,
                 url='https://github.com/phageghost/PEAS',
                 author=AUTHOR,
                 author_email='peas@phageghost.net',
                 license='MIT',
                 packages=['peas'],
                 install_requires=['numpy', 'datetime', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'empdist'],
                 zip_safe=False,
                 classifiers=(
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ),
                 )
