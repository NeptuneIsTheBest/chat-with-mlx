from setuptools import setup, find_packages

setup(
    name="chat-with-mlx-refresh",
    version="0.1.25",
    author="NeptuneIsTheBest",
    author_email="13058097081a@gmail.com",
    description="An all-in-one chat Web UI based on the MLX framework, designed for Apple Silicon.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="."),
    url="https://github.com/NeptuneIsTheBest/chat-with-mlx",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").read().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "chat-with-mlx=chat_with_mlx_refresh.app:main",
        ]
    }
)
