from setuptools import setup, find_packages

setup(
    name="stratvector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "numba>=0.57.0",
        "ib_insync>=0.9.86",
        "plotly>=5.18.0",
        "toml>=0.10.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ],
    },
    python_requires=">=3.10",
    author="Your Name",
    author_email="your.email@example.com",
    description="A quantitative trading strategy framework",
    long_description=open("README.md").read() if open("README.md").readable() else "",
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
) 