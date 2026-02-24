from setuptools import setup, find_packages

setup(
    name='bdpn',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    version='0.1.15',
    description='Estimation of BDPN parameters from phylogenetic trees.',
    author='Anna Zhukova',
    author_email='anna.zhukova@pasteur.fr',
    url='https://github.com/evolbioinfo/bdpn',
    keywords=['phylogenetics', 'birth-death model', 'partner notification'],
    install_requires=['six', 'ete3', 'numpy', "scipy==1.10.0-rc1", 'biopython'],
    entry_points={
            'console_scripts': [
                'bdpn_infer = bdpn.bdpn_model:main',
                'bd_infer = bdpn.bd_model:main',
                'bdmult_infer = bdpn.bdmult_model:main',
                'bdssmult_infer = bdpn.bdssmult_model:main',
                'bdpn_loglikelihood = bdpn.bdpn_model:loglikelihood_main',
                'bd_loglikelihood = bdpn.bd_model:loglikelihood_main',
                'bdmult_loglikelihood = bdpn.bdmult_model:loglikelihood_main',
                'bdssmult_loglikelihood = bdpn.bdssmult_model:loglikelihood_main',
                'pn_test = bdpn.model_distinguisher:main',
            ]
    },
)
