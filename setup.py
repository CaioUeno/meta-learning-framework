import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meta_learning_framework_CAIO_UENO", # Replace with your own username
    version="0.0.1",
    author="Caio Ueno",
    author_email="caiol.ueno@gmail.com",
    description="A framework to apply a meta learning algorithm (learn how to ensemble models)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CaioUeno/meta-learning-framework/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Freely Distributable",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)