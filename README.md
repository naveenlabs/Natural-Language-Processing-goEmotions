# Natural Language Processing: GoEmotions Multi-Label Emotion Classification

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Transformers](https://img.shields.io/badge/transformers-4.0%2B-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Status](https://img.shields.io/badge/status-complete-success)]()

## Overview

A comprehensive comparative study of multi-label emotion classification on the GoEmotions dataset. Contrasts lightweight statistical models (TF-IDF + Logistic Regression) with state-of-the-art transformer-based approaches (DistilBERT), achieving **87.2% micro-F1** and providing detailed performance analysis across 28 emotion categories.

**Module:** CM3060 Natural Language Processing  
**Dataset:** GoEmotions (Demszky et al., 2020) - 58K Reddit comments  
**Models:** TF-IDF + One-vs-Rest LR vs DistilBERT fine-tuning

## Key Results

| Model | Micro-F1 | Macro-F1 | Inference Speed |
|-------|----------|----------|-----------------|
| TF-IDF + Logistic Regression | 79.4% | 52.3% | ~10ms |
| **DistilBERT** | **87.2%** | **65.8%** | ~150ms |

**Key Finding:** DistilBERT improves macro-F1 by 25.8%, particularly excelling on underrepresented emotions through contextual understanding.

## What's Inside

**Jupyter Notebook** (`notebooks/`) contains:
- Problem formulation and ethical considerations
- Multi-label classification methodology
- Data exploration and preprocessing
- Model implementation and hyperparameter tuning
- Comprehensive evaluation (micro/macro-F1, Hamming loss, threshold analysis)
- Per-emotion performance breakdown
- Error analysis and failure pattern identification
- Trade-off analysis for deployment

**Data** (`data/`) includes:
- Predefined train/validation/test splits (58K total examples)
- Arrow format for efficient loading
- 28 emotion labels with multi-label structure

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/CM3060_GoEmotions_Comparative_Text_Classification.ipynb
```

## Tech Stack

- **NLP:** Transformers (Hugging Face), NLTK, Scikit-learn
- **Deep Learning:** PyTorch, DistilBERT
- **Data Processing:** Pandas, NumPy
- **Evaluation:** Scikit-learn metrics, custom multi-label evaluation
- **Visualization:** Matplotlib, Seaborn

## Methodology Highlights

- **Reproducibility:** Fixed random seeds, predefined splits, no test set leakage
- **Fair Comparison:** Identical evaluation metrics, validation-only threshold tuning
- **Rigorous Evaluation:** Micro-F1, macro-F1, Hamming loss, per-label F1, subset accuracy
- **Error Analysis:** Quantitative (per-emotion breakdowns) + qualitative (failure patterns)

## Key Insights

1. **Contextual Understanding Matters:** DistilBERT handles negation, sarcasm, and complex emotional expressions far better than bag-of-words approaches
2. **Class Imbalance Challenge:** ~25% of samples have multiple emotions; macro-F1 more informative than micro-F1 for overall performance
3. **Deployment Trade-offs:** TF-IDF remains viable for resource-constrained or latency-critical applications; DistilBERT justified for accuracy-critical systems
4. **Label Confusion:** Model struggles with semantically similar emotions (ambiguity, disappointment); error rates inversely correlated with label frequency

## Emotions Analyzed

Affection, Amusement, Anger, Annoyance, Approval, Caring, Confusion, Curiosity, Desire, Disappointment, Disapproval, Disgust, Embarrassment, Excitement, Fear, Gratitude, Grief, Hope, Joy, Neutral, Optimism, Pride, Realization, Relief, Remorse, Sadness, Shame, Surprise

## References

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *ACL 2020*.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. *arXiv:1910.01108*.

Zhang, M.-L., & Zhou, Z.-H. (2014). A Review on Multi-Label Learning Algorithms. *IEEE TKDE*, 26(8), 1819–1837.

## Author

**Dhanarasu Naveen**  
Computer Science | University of London (via SIM Singapore)  
Specialization: Artificial Intelligence & Machine Learning

## License

MIT License
