import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='CCLR_LPNO',
        version='0.1',
        description='Coupled-Cluster Linear Response with Local Correlation',
        author='Ruhee Dcunha',
        packages=['ccsd_lpno',],
        install_requires=['numpy>=1.7',],
        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
                'tox',],
        },
        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
            'tox',
        ],
        long_description=open('README.txt').read(),
)
