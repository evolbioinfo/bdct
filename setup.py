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
    version='0.1.23',
    description='Maximum likelihood estimation of BD and BD-CT(1) parameters from phylogenetic trees.',
    author='Anna Zhukova',
    author_email='anna.zhukova@pasteur.fr',
    url='https://github.com/evolbioinfo/bdct',
    keywords=['phylogenetics', 'birth-death model', 'partner notification', 'contact tracing', 'BD', 'BD-CT'],
    install_requires=['six', 'ete3', 'numpy==2.0.2', "scipy==1.14.1", 'biopython'],
    entry_points={
            'console_scripts': [
                'bdct_infer = bdct.bdct_model:main',
                'bd_infer = bdct.bd_model:main',
                'bdct_loglikelihood = bdct.bdct_model:loglikelihood_main',
                'bd_loglikelihood = bdct.bd_model:loglikelihood_main',
                'ct_test = bdct.model_distinguisher:main',
            ]
    },
)
