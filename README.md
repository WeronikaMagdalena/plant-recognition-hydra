# Plant Recognition — Multi-Head Architecture (Hydra) 🌿

A plant species recognition system using a shared ResNet50 backbone and three different classifier heads:  
- **Decision Tree**  
- **Support Vector Machine (SVM)**  
- **Artificial Neural Network (ANN)**  

This architecture allows each team member to design and evaluate their own classifier head while using a common feature extractor.

---

## Project Overview

This project performs **plant species classification** using a hybrid approach:

1. **ResNet50** is used as a fixed **feature extractor**, producing 2048-dim embeddings for each plant image.
2. These embeddings serve as input to three independent classifier heads:
   - Decision Tree (DT)
   - Support Vector Machine (SVM)
   - Artificial Neural Network (ANN)
3. Each classifier is trained, evaluated, and compared on the same embedding dataset.

This modular architecture is nicknamed **Hydra** because of its single backbone and multiple heads.

---

## Repository Structure

```
plant-recognition-hydra/
│
├── embeddings/
│   └── info.txt
│
├── models/
│   ├── backbone/           
│   │   └── resnet50.py
│   ├── decision_tree/
│   ├── svm/
│   └── ann/
│
├── presentation/
│
├── requirements.txt
│
└── README.md
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

## Contributors

| Student | Component |
|--------|-----------|
| Weronika Wójcik | Decision Tree Head |
| Filip Skibiński | SVM Head |
| Adam Wielogórski | ANN Head |








