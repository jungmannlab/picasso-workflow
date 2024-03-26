from setuptools import setup, find_packages

classifiers = [
    "Development Status :: Development",
    "Intended Audience :: Education",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

setup(
    name="picasso-workflow",
    version="0.0.1",
    description="Automation and documentation of DNA-PAINT analysis workflows",
    long_description=(
        open("README.md").read() + "\n\n" + open("CHANGELOG.txt").read()
    ),
    url="",
    # author='MPG - Heinrich Grabmayr',
    # author_email='hgrabmayr@biochem.mpg.de',
    license="MIT",
    classifiers=classifiers,
    keywords="picasso",
    packages=find_packages(),
)
