import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xrbeall",
    version="0.0.1",
    author="Ondrej Grover",
    author_email="ondrej.grover@gmail.com",
    description="Beall (1982) wavenumber-frequency spectra calculation in xarray",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smartass101/xrbeall",
    py_modules=['xrbeall'],
    install_requires=['xarray', 'xrscipy'],
)
