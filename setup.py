from setuptools import setup, find_packages


setup(
    name="spytnik",
    version="0.1.0",
    description="Deep learning field of experiments",
    url="https://github.com/MikhailKravets/Spytnik",
    author="Mikhail Kravets",
    author_email="michkravets@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    keywords="deep-learning neural-networks research science",
    python_requires=">=3.6",
    packages=find_packages()
)