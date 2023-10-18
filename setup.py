from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    # TODO: This is a hack to get around the fact that we can't read the requirements.txt file, we should fix this.
    # def _read_reqs(relpath):
    #     fullpath = os.path.join(Path(__file__).parent, relpath)
    #     with open(fullpath) as f:
    #         return [
    #             s.strip()
    #             for s in f.readlines()
    #             if (s.strip() and not s.startswith("#"))
    #         ]

    REQUIREMENTS = [
        "einops",
        "einops-exts",
        "transformers",
        "torch",
        "torchvision",
        "pillow",
        "more-itertools",
        "datasets",
        "braceexpand",
        "webdataset",
        "wandb",
        "nltk",
        "scipy",
        "inflection",
        "sentencepiece",
        "open_clip_torch",
    ]

    setup(
        name="open_flamingo",
        packages=find_packages(),
        include_package_data=True,
        version="0.0.2",
        license="MIT",
        description="An open-source framework for training large multimodal models",
        long_description=long_description,
        long_description_content_type="text/markdown",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning"],
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.9",
        ],
    )
