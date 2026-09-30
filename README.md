# small-data-healthcare-transformer
Small-data transformer for synthetic healthcare data generation and evaluation
The same general approach was applied to two healthcare datasets:

- Synthea
- MIMIC-IV
  
For each dataset, models were trained using populations of 100, 1,000, and 10,000 patients. Each trained model was then used to generate 5,000 synthetic patient records.
The generated populations were evaluated across three areas:
- Privacy and memorization risk
- Population representation
- Structural fidelity and diversity
The transformer is small and relatively simple. Just a single transformer with 2-heads and MLP layer. The goal was not to build the best possible synthetic-data generator, but to examine how the amount of available training data affects the generated population while keeping the model architecture fixed.

## Repository Structure

`data_*.py` - data preparation and tokenization
`model_*.py` - transformer architecture
`train_*.py` - model training
`generate_*.py` - synthetic patient generation
 `*_compare.py` - evaluation and comparison

## Data
The Synthea and MIMIC-IV pipelines are kept separate but follow the same general experimental approach.
Synthea provides fully synthetic patient records.
MIMIC-IV is a de-identified clinical dataset. The original MIMIC-IV data are not included in this repository.
Python and PyTorch were used for data processing, model development, training, generation, and evaluation.

## Project
University of Texas at Austin  
MS Data Science  
AI in Healthcare
