# Bias Mitigation in DistilBERT: A Systematic Fine-Tuning Approach

**Author:** Erfan Esfahanian  
**Course:** Natural Language Processing - Università degli Studi di Milano  
**Professor:** Alfio Ferrara

## Project Overview

This project investigates systematic bias reduction in DistilBERT through targeted fine-tuning approaches. We developed and evaluated multiple bias mitigation strategies, achieving a **78.7% reduction** in stereotypical preferences on the StereoSet benchmark.

## Key Results

- **Baseline Bias Score:** +0.0234
- **Final Bias Score:** +0.0050  
- **Bias Reduction:** 78.7%
- **Evaluation Dataset:** 2,106 StereoSet examples across 4 bias categories

## Methodology Summary

### Phase 1: Bias Detection and Baseline Evaluation
- Implemented masked language modeling for bias detection
- Discovered systematic profession-gender associations (nurse: 6.27% "she" vs 1.45% "he")
- Established comprehensive StereoSet evaluation pipeline
- Identified bias patterns across profession, race, gender, and religion categories

### Phase 2: Initial Fine-Tuning Attempt (Failed)
- **Training Data:** 12 examples (0.2% coverage)
- **Parameters:** Aggressive (learning rate: 5e-5, 5 epochs)
- **Result:** Bias INCREASED from +0.0234 to +0.0243
- **Cause:** Overcorrection, insufficient scale, memorization vs. generalization

### Phase 3: Strategic Redesign and Success
- **Training Data:** 132 examples (2.1% coverage, 11x increase)
- **Parameters:** Conservative (learning rate: 2e-5, 3 epochs)
- **Approach:** Balanced counter-stereotypical examples across professions
- **Result:** 78.7% bias reduction achieved

## Technical Implementation

### Bias Detection Methods
```python
# Masked Language Modeling Probability Analysis
def get_sentence_probability(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    return np.mean(token_log_probs)

# Fill-Mask Pipeline Testing
fill_mask = pipeline('fill-mask', model='distilbert-base-uncased')
results = fill_mask("The nurse was tired, so [MASK] went home early.")
```

### Training Data Examples
```
"The nurse was decisive when he made the treatment plan."
"The engineer was empathetic as she listened to concerns."
"The secretary was analytical while he reviewed data."
```

## Repository Structure

```
├── main.ipynb                           # Complete project workflow
├── expanded_counter_bias_data.json     # 132 training examples
├── day9_improved_strategy.json         # Training configuration
├── fine_tuned_bias_results.json        # Evaluation results
├── day8_analysis_results.json          # Failure analysis
├── improved-counter-bias-model-final/  # Final debiased model
└── README.md                           # This file
```

## Key Findings

### What Worked
- **Scale Matters:** 11x increase in training data was critical for success
- **Conservative Training:** Lower learning rate (2e-5) prevented overcorrection
- **Balanced Examples:** Equal gender representation across professions
- **Systematic Evaluation:** Comprehensive StereoSet assessment revealed true impact

### What Failed
- **Small Datasets:** 12 examples insufficient for global bias reduction
- **Aggressive Parameters:** High learning rate caused extreme local overcorrection
- **Narrow Focus:** Single-category training ignored broader bias patterns

### Bias Responsiveness
- **High Response:** Engineering profession (near-perfect balance achieved)
- **Moderate Response:** Secretary profession (significant improvement)
- **Low Response:** Nursing profession (persistent bias despite training)

## Research Contributions

1. **Systematic Methodology:** Reproducible framework for bias detection and mitigation
2. **Scale Requirements:** Empirical evidence for training data coverage thresholds
3. **Failure Documentation:** Instructive analysis of overcorrection patterns
4. **Practical Results:** Deployable bias-reduced model with measurable improvements

## Usage

### Loading the Debiased Model
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('./improved-counter-bias-model-final')
model = AutoModelForMaskedLM.from_pretrained('./improved-counter-bias-model-final')
```

### Running Bias Evaluation
```python
# Load StereoSet data and evaluate bias scores
# See main.ipynb for complete evaluation pipeline
```

## Dependencies

```
transformers==4.46.2
torch==2.7.0
datasets==3.1.0
scikit-learn==1.2.2
numpy==1.26.4
pandas==2.1.4
matplotlib==3.8.0
```

## Results Summary

| Metric | Baseline | Improved | Change |
|--------|----------|----------|---------|
| Stereotype Score | -12.9418 | -12.8062 | +0.1356 |
| Anti-Stereotype Score | -12.9651 | -12.8112 | +0.1539 |
| **Bias Score** | **+0.0234** | **+0.0050** | **-0.0184** |
| **Bias Reduction** | - | - | **78.7%** |

## Academic Paper

Complete technical documentation available in the accompanying research paper: "Systematic Bias Mitigation in DistilBERT through Targeted Fine-Tuning: A Comprehensive Study of Scale, Strategy, and Evaluation"

## License

This project is for academic purposes as part of coursework at Università degli Studi di Milano.

## Acknowledgments

- Professor Alfio Ferrara for project guidance
- StereoSet benchmark developers for evaluation framework
- Hugging Face for transformer model implementations