import os

from setuptools import setup, find_packages

setup(
    name='bdct',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={'bdct': [os.path.join('..', 'README.md'),
                            os.path.join('..', 'LICENCE')]},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    version='0.1.27',
    description='Maximum likelihood estimation of BD, BD-CT(1) and BDSKY parameters from phylogenetic trees.',
    author='Anna Zhukova',
    author_email='anna.zhukova@pasteur.fr',
    url='https://github.com/evolbioinfo/bdct',
    keywords=['phylogenetics', 'birth-death model', 'birth-death skyline', 'partner notification', 'contact tracing', 'BD', 'BD-CT', 'BDSKY'],
    install_requires=['six', 'ete3', 'numpy', 'scipy', 'sympy', 'biopython'],
    requires=['six', 'ete3', 'numpy', 'scipy', 'sympy', 'biopython'],
    entry_points={
            'console_scripts': [
                'bdct_infer = bdct.bdct_model:main',
                'bd_infer = bdct.bd_model:main',
                'bdsky_infer = bdct.bdsky_model:main',
                'bdct_loglikelihood = bdct.bdct_model:loglikelihood_main',
                'bd_loglikelihood = bdct.bd_model:loglikelihood_main',
                'bdsky_loglikelihood = bdct.bdsky_model:loglikelihood_main',
                'ct_test = bdct.model_distinguisher:main',
            ]
    },
)
