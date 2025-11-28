# Plant Recognition — Multi-Head Architecture (Hydra) 🌿

A plant species recognition system using a shared ResNet50 backbone and three different classifier heads:  
- **Decision Tree**  
- **Support Vector Machine (SVM)**  
- **Multi-Layer Perceptron (MLP)**  

This architecture allows each team member to design and evaluate their own classifier head while using a common feature extractor.

---

## Project Overview

This project performs **plant species classification** using a hybrid approach:

1. **ResNet50** is used as a fixed **feature extractor**, producing 2048-dim embeddings for each plant image.
2. These embeddings serve as input to three independent classifier heads:
   - Decision Tree (DT)
   - Support Vector Machine (SVM)
   - Multi-Layer Perceptron (MLP)
3. Each classifier is trained, evaluated, and compared on the same embedding dataset.

This modular architecture is nicknamed **Hydra** because of its single backbone and multiple heads.

---

## Repository Structure

```
plant-recognition-multihead/
│
├── embeddings/
│   ├── train/                
│   ├── test/
│   └── info.txt
│
├── models/
│   ├── backbone/           
│   │   └── resnet50.py
│   ├── decision_tree/
│   ├── svm/
│   └── mlp/
│
├── notebooks/
│
├── results/
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/WeronikaMagdalena/plant-recognition-hydra.git
cd plant-recognition-hydra
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Feature Extraction (ResNet50)

Run this script to generate the **2048-dim embeddings**:

```bash
python scripts/extract_embeddings.py
```

This will populate:

```
embeddings/train/
embeddings/test/
```

These embeddings are used by all three classifier heads.

---

## Classifier Heads

### Decision Tree  
Developed by **Weronika Wójcik**  
Training script:

```bash
python models/decision_tree/train_dt.py
```

### SVM  
Developed by **Filip Skibidiński**

### MLP  
Developed by **Adam Wielogórski**

Each head reads the same embeddings and outputs its own model + metrics.

---

## Results

All results, plots, metrics, and comparisons are stored in:

```
results/
```

Subfolders:

- `dt/`
- `svm/`
- `mlp/`
- `comparisons/`

---

## Notebooks

Exploration, visualizations, and experiments are in:

```
notebooks/
```

---

## Contributors

| Student | Component |
|--------|-----------|
| Weronika Wójcik | Decision Tree Head |
| Filip Skibiński | SVM Head |
| Adam Wielogórski | MLP Head |

---

## License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

## Acknowledgements

- ResNet50 implementation from PyTorch / TensorFlow
- Original plant dataset ([dataset source](https://www.kaggle.com/datasets/datajameson/planclef
)<img width="1067" height="88" alt="image" src="https://github.com/user-attachments/assets/a4d09f1f-d853-4e0b-9441-480a49666f6d" />
)
- Hydra nickname inspired by the multi-head architecture





