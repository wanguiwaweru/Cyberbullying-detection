# Cyberbullying Detection

A machine learning system for detecting whether comments posted have harmful sentiments.


## ⚙️ Features
- sentiment classification under 7 categories.
- classification confidence level

## Built With 🛠

- [Fast API](https://fastapi.tiangolo.com/) - An open source modern, high-performance web framework for building APIs with Python based on standard type hints.
- [Google Colab](https://colab.research.google.com/) -  Colab is a free Jupyter notebook environment that allows one to write and execute python code through the browser, and is well suited to machine learning, and data analysis. The platform offers access to free GPU and was used for model training.

- [Git LFS](https://git-lfs.com/)- An open source Git extension for versioning large files.
- [BERT model](https://huggingface.co/docs/transformers/model_doc/bert) - An open source machine learning framework for natural language processing (NLP). The BERT model was fine-tuned turing training.


## Getting Started 

### Prerequisites
[Python 3.9](https://www.python.org/downloads/)
[Git LFS](https://git-lfs.com/)


To get started, clone the repo and run: `pip install -r requirements.txt`

On your terminal run `uvicorn main:app`

Navigate to http://127.0.0.1:8000/docs to view the interactive docs.

On postman create a POST request to http://127.0.0.1:8000/api and add a request body that contains the text and the request id.
