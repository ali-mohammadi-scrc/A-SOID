from setuptools import setup, find_packages
from os import path
curr_dir = path.abspath(path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='asoid',
    version='0.3',
    description='ASOiD: An active learning approach to behavioral classification',
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/YttriLab/A-SOID",
        "Bug Tracker": "https://github.com/YttriLab/A-SOID/issues"
    },
    url="https://github.com/YttriLab/A-SOID",
    author=['Jens F. Schweihoff','Alexander Hsu'],
    entry_points={
        "console_scripts": [
                            "asoid =asoid.__main__:main"
                            ]
        },
    packages=find_packages(),  # same as name
    include_package_data=True,
    install_requires=["matplotlib"
                    ,"numpy"
                    ,"pandas"
                    ,"seaborn"
                    ,"streamlit>=1.20.0"
                    ,"streamlit_option_menu"
                    ,"hydralit"
                    ,"opencv-python"
                    ,"tqdm"
                    ,"stqdm"
                    ,"scikit-learn<1.3.0"
                    ,"h5py"
                    ,"plotly"
                    ,"pillow"
                    ,"joblib==1.3.2"
                    ,"scipy"
                    ,"ipython"
                    ,"psutil"
                    ,"numba"
                    ,"hdbscan==0.8.33"
                    ,"setuptools"
                    ,"umap-learn"
                    ,"click"
                    ,"moviepy"
                    ,"ffmpeg-python"
                    ],
    python_requires=">=3.7",
)
