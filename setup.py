from setuptools import setup, find_packages

setup(
    name='autoroute-manager',
    version='0.1.0',
    description='Python interface for working with AutoRoute',
    keywords='AutoRoute',
    author='Louis Ricky Rosas',
    author_email='louisrickyrosas@gmail.com',
    url='https://github.com/erdc/AutoRoutePy',
    download_url='https://github.com/erdc/AutoRoutePy/archive/2.1.0.tar.gz',
    license='BSD 3-Clause',
    packages=find_packages(),
    install_requires=['gdal'],
    classifiers=[
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.10',
            ],
)