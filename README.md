# 🍷 Wine Classification Using a Decision Tree Classifier

A machine learning project that applies a **Decision Tree Classifier** (Scikit-learn) to the classic Wine dataset to predict which of three grape cultivars a wine belongs to, based on its chemical properties.

---

## 📌 Overview

The goal is to classify wines into **Class 1, 2, or 3** using 13 chemical features. The notebook walks through the full workflow: data inspection, train-test split, model building, evaluation, and visualization of the trained tree.

## 📂 Project Structure

```
├── Decision_Tree_Algorithm_WineDataset.ipynb   # Main notebook
├── wine_data.csv                               # Dataset
└── README.md
```

## 📊 Dataset

- **Samples:** 178
- **Columns:** 14 (1 target + 13 features)
- **Missing values:** None
- **Duplicates:** None

| Type | Columns |
|------|---------|
| **Target** | `Class` (1, 2, 3) |
| **Features** | Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines, Proline |

## ⚙️ Methodology

1. **Load and inspect** the data (shape, dtypes, null values, duplicates)
2. **Separate** features (`x`) and target (`y`)
3. **Split** into 80% training / 20% testing (`random_state=42`)
4. **Train** a `DecisionTreeClassifier` with:
   - `criterion='entropy'`
   - `max_leaf_nodes=10` (limits tree complexity to reduce overfitting)
5. **Predict** on the test set
6. **Evaluate** using accuracy, confusion matrix, and classification report
7. **Visualize** the tree with `plot_tree`

## 📈 Results

**Test accuracy: ~91.7%**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 1 | 0.93 | 1.00 | 0.97 | 14 |
| 2 | 0.88 | 1.00 | 0.93 | 14 |
| 3 | 1.00 | 0.62 | 0.77 | 8 |

**Confusion matrix**

```
[[14  0  0]
 [ 0 14  0]
 [ 1  2  5]]
```

**Key observations**
- Classes 1 and 2 were classified perfectly (recall = 1.00).
- Class 3 was the hardest to predict: 3 of 8 samples were misclassified.
- The tree's top splits used **OD280/OD315**, **Color intensity**, **Flavanoids**, and **Alcohol**, showing these are strong indicators of wine class.

## 🛠️ Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install dependencies
   ```bash
   pip install numpy pandas scikit-learn matplotlib jupyter
   ```
3. Make sure the dataset filename matches the one in the notebook (`pd.read_csv("wine.data.csv")` vs. `wine_data.csv`), then launch the notebook
   ```bash
   jupyter notebook Decision_Tree_Algorithm_WineDataset.ipynb
   ```

## 🔮 Future Improvements

- Tune hyperparameters (`max_depth`, `min_samples_split`) with GridSearchCV
- Add cross-validation for a more reliable accuracy estimate
- Plot feature importances
- Compare with Random Forest and other classifiers
- Improve Class 3 recall (e.g., class weighting)

## 📚 Reference

Dataset: [UCI Machine Learning Repository – Wine Data Set](https://archive.ics.uci.edu/dataset/109/wine)

## 👤 Author

**Shobana C**
[GitHub]https://github.com/shobanachandran24-lab · [[LinkedIn](https://www.linkedin.com/in/shobanachandran2402/)]

---

⭐ If you found this helpful, consider giving the repo a star!
