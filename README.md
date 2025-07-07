<div align="left" style="position: relative;">
<img src="https://img.icons8.com/?size=512&id=55494&format=png" align="right" width="30%" style="margin: -20px 0 0 20px;">
<h1>BREAST_CANCER_CLASSIFICATION</h1>
<p align="left">
	<em><code>❯ REPLACE-ME</code></em>
</p>
<p align="left">
	<img src="https://img.shields.io/github/license/Trippod/Breast_Cancer_Classification?style=plastic&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Trippod/Breast_Cancer_Classification?style=plastic&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Trippod/Breast_Cancer_Classification?style=plastic&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Trippod/Breast_Cancer_Classification?style=plastic&color=0080ff" alt="repo-language-count">
</p>
<p align="left">Built with the tools and technologies:</p>
<p align="left">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=plastic&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=plastic&logo=FastAPI&logoColor=white" alt="FastAPI">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=plastic&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=plastic&logo=Docker&logoColor=white" alt="Docker">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=plastic&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=plastic&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=plastic&logo=Pydantic&logoColor=white" alt="Pydantic">
</p>
</div>
<br clear="right">

## 🔗 Quick Links

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
  - [📂 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)
  - [🤖 Usage](#🤖-usage)
  - [🧪 Testing](#🧪-testing)
- [📌 Project Roadmap](#-project-roadmap)
- [🔰 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

Repozytorium prezentuje kompletny pipeline ML do klasyfikacji nowotworu piersi (zbiór **Breast Cancer Wisconsin**):  
1. Eksploracja i wizualizacja danych (EDA)  
2. Przygotowanie cech i skaler  
3. Trening pięciu klasycznych modeli (Logistic, Tree, Forest, SVM, KNN) i MLP  
4. Ujednolicony router predykcji → jeden endpoint `/predict`  
5. Udostępnienie modelu w FastAPI, konteneryzacja w Docker (`python:3.11-slim`).


---

## 👾 Features

* Sześć modeli (Logistic, Tree, Forest, SVM, KNN, Neural Net)  
* Router ładuje wszystkie modele + scaler przy starcie → natychmiastowa odpowiedź  
* FastAPI + Swagger UI  
* Lekki obraz `python:3.11-slim` (≈450 MB) – `docker run -p 8000:8000 cancer-api`  
* Metryki (Accuracy / F1 / AUC) i heat-mapy w **outputs/**  

---

## 📁 Project Structure

```sh
└── Breast_Cancer_Classification/
    ├── Dockerfile
    ├── requirements.txt
    └── venv
        ├── models
        │   ├── forest.pkl
        │   ├── knn.pkl
        │   ├── logistic.pkl
        │   ├── nn.h5
        │   ├── scaler.pkl
        │   ├── svm.pkl
        │   └── tree.pkl
        └── src
            ├── __init__.py
            ├── data_preprocessing.py
            ├── data_visualization.py
            ├── main.py
            ├── model_cnn.py
            ├── model_forest.py
            ├── model_knn.py
            ├── model_logistic_reg.py
            ├── model_tree.py
            ├── models_svm.py
            └── predicted_router.py
```


### 📂 Project Index
<details open>
	<summary><b><code>BREAST_CANCER_CLASSIFICATION/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/Dockerfile'>Dockerfile</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- venv Submodule -->
		<summary><b>venv</b></summary>
		<blockquote>
			<details>
				<summary><b>src</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/model_logistic_reg.py'>model_logistic_reg.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/model_forest.py'>model_forest.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/main.py'>main.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/data_visualization.py'>data_visualization.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/model_knn.py'>model_knn.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/models_svm.py'>models_svm.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/predicted_router.py'>predicted_router.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/model_tree.py'>model_tree.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/data_preprocessing.py'>data_preprocessing.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Trippod/Breast_Cancer_Classification/blob/master/venv/src/model_cnn.py'>model_cnn.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with Breast_Cancer_Classification, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip
- **Container Runtime:** Docker


### ⚙️ Installation

Install Breast_Cancer_Classification using one of the following methods:

**Build from source:**

1. Clone the Breast_Cancer_Classification repository:
```sh
❯ git clone https://github.com/Trippod/Breast_Cancer_Classification
```

2. Navigate to the project directory:
```sh
❯ cd Breast_Cancer_Classification
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```


**Using `docker`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/)

```sh
❯ docker build -t Trippod/Breast_Cancer_Classification .
```




### 🤖 Usage
Run Breast_Cancer_Classification using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```


**Using `docker`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/)

```sh
❯ docker run -it {image_name}
```


### 🧪 Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pytest
```


## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/Trippod/Breast_Cancer_Classification/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/Trippod/Breast_Cancer_Classification/issues)**: Submit bugs found or log feature requests for the `Breast_Cancer_Classification` project.
- **💡 [Submit Pull Requests](https://github.com/Trippod/Breast_Cancer_Classification/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/Trippod/Breast_Cancer_Classification
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/Trippod/Breast_Cancer_Classification/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Trippod/Breast_Cancer_Classification">
   </a>
</p>
</details>

---

## 🎗 License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## 🙌 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---