# GAIA — Pediatric Meal Planning System

**GAIA** (Gerador Automático de Intervenção Alimentar) is a multi-agent system that generates
nutritionally adequate, affordable, and age-appropriate meal plans for children aged 0–59 months
in food-insecure environments.

Developed as a final undergraduate thesis (TCC) at UNISINOS, 2026.

---

## Pipeline Overview

The system runs five sequential agents:

1. **Assessment Agent** — anthropometric z-score computation (WHO LMS method) and DRI derivation
2. **Food Retriever Agent** — food preference scoring via a LightGBM model trained on 24-hour dietary recall data
3. **Meal Planner Agent** — quantity allocation via Linear Programming with DRI-based nutritional constraints
4. **Validation Module** — nutritional adequacy check against WHO/DRI thresholds
5. **Report Agent** — structured report with optional LLM-generated caregiver narrative (Gemini)

---

## Data Sources

| Source | Description | Usage |
|--------|-------------|-------|
| TACO | Brazilian Food Composition Table (UNICAMP) | Nutritional composition |
| TBCA | Brazilian Food Composition Database (USP) | Nutritional composition |
| ENANI-2019 | National Children's Nutrition Survey | Anthropometric profiles + dietary recall for ML training |
| WHO LMS Tables | WHO Child Growth Standards | Z-score computation |
| DRI (IOM/NIH) | Dietary Reference Intakes | Nutritional targets |

> **Note:** The ENANI datasets (`enani_profiles.csv`, `enani_recall_24h_clean.csv`) are not included
> in this repository due to data governance restrictions. Contact UFRJ/ENANI for access:
> https://enani.nutrition.ufrj.br

> **Note:** The trained preference model (`models/preference_model.pkl`) is not included in this
> repository because training requires the restricted ENANI data. See
> [Training the Preference Model](#training-the-preference-model) below.

---

## Requirements

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

---

## Setup

```bash
git clone https://github.com/silvaadriele/GAIA_Generative_AI_for_Affordable_Nutrition.git
cd GAIA_Generative_AI_for_Affordable_Nutrition
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Pipeline

### Option 1 — JSON profile file (recommended)

Create a child profile file (e.g., `perfil.json`):

```json
{
  "age_months": 18,
  "sex": "F",
  "weight_kg": 9.8,
  "height_cm": 78.5,
  "region": "Nordeste"
}
```

`sex` accepts: `M` or `F`  
`region` accepts: `Norte`, `Nordeste`, `Centro-Oeste`, `Sudeste`, `Sul` (optional)  
`height_cm` is optional for children under 6 months

Run:

```bash
python main.py perfil.json
```

The report is printed to the terminal and saved automatically to `evaluation/results/`.

### Option 2 — Command-line flags

```bash
python -m orchestrator.gaia --age 18 --sex F --weight 9.8 --height 78.5 --region Nordeste
```

Available flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--age` | Age in months (required) | — |
| `--sex` | Sex: `M` or `F` (required) | — |
| `--weight` | Weight in kg (required) | — |
| `--height` | Height in cm | `None` |
| `--region` | Brazilian region | `None` |
| `--top-k` | Number of food candidates for LP | `50` |
| `--no-gemini` | Disable LLM caregiver narrative | off |
| `--out` | Output directory | `evaluation/results/` |

---

## Enabling the Gemini API (caregiver narrative)

By default, `main.py` runs without the LLM step. To enable the AI-generated caregiver narrative:

1. Obtain a Gemini API key at https://aistudio.google.com
2. Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

3. In `main.py`, change `use_gemini=False` to `use_gemini=True`:

```python
result = GAIA(use_gemini=True).run(profile)
```

When using Option 2 (CLI flags), Gemini is **enabled by default**. Pass `--no-gemini` to disable it.

---

## Training the Preference Model

The trained model (`models/preference_model.pkl`) is not distributed with this repository.
To train it from scratch, you will need the ENANI-2019 datasets (see [Data Sources](#data-sources)).

Once you have the data:

```bash
# 1. Place the ENANI files in data/raw/
#    - enani_profiles.csv
#    - enani_recall_24h_clean.csv

# 2. Preprocess the food database
python scripts/preprocess_foods.py

# 3. Train the LightGBM preference model
python scripts/train_preference_model.py
```

The trained model is saved to `models/preference_model.pkl`.

---

## Project Structure

```
GAIA/
├── agents/              # AssessmentAgent, FoodRetrieverAgent, MealPlannerAgent, ReportAgent
├── modules/             # LP solver, nutritional validator, food price estimator
├── orchestrator/        # GAIA pipeline orchestrator (also exposes CLI)
├── scripts/             # Data preprocessing and model training scripts
├── prompts/             # Prompt templates for the Report Agent
├── data/
│   ├── gaia_food.db     # Unified food database (TACO + TBCA)
│   ├── processed/       # Preprocessed food composition tables
│   ├── reference/       # DRI tables, WHO LMS tables, TACO mapping
│   └── raw/             # ENANI datasets (not included — see Data Sources)
├── models/
│   ├── preference_model.pkl   # Trained LightGBM model (not included — see above)
│   └── feature_importance.csv
├── evaluation/
│   ├── system_eval/     # 50-profile system evaluation reports
│   ├── appendix_final/  # Appendix meal plan examples
│   └── results/         # Generated reports (created at runtime, not tracked)
├── tests/
├── main.py              # Entry point (JSON profile input)
├── perfil_teste.json    # Example child profile
└── requirements.txt
```

---

## Citation

This system was developed as a final undergraduate thesis at UNISINOS (2026).
If you use this work, please cite accordingly.
