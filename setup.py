import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='cclr',
        version='0.1',
        description='Coupled-Cluster Linear Response with Local Correlation',
        author='Ruhee Dcunha',
        packages=setuptools.find_packages(),
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
