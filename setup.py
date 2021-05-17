import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meta-learning-framework",
    version="0.0.1",
    author="Caio Ueno",
    author_email="caiol.ueno@gmail.com",
    description="A framework to apply a meta learning algorithm at instance level (learn how to ensemble models)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CaioUeno/meta-learning-framework/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)