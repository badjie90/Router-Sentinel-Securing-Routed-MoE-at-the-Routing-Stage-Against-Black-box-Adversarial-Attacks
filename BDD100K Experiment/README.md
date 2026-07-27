# Router-Sentinel-Securing-Routed-MoE-at-the-Routing-Stage-Against-Black-box-Adversarial-Attacks
Router-Sentinel: Securing Routed Mixture-of-Experts at the Routing Stage Against Black-box Adversarial Attacks


# Adversarial Robustification of a BDD100K MoE Router

This repository trains and evaluates a robustified router for a
condition-specialized BDD100K Mixture-of-Experts (MoE) model. It builds on a
previously trained baseline model and adds adversarial training and a sentinel
mechanism intended to identify or handle unreliable routing under attack.

## Files in this repository

```text
.
├── README.md
├── requirements.txt
└── scripts/
    ├── bdd100k_moe_train.py
    ├── bdd100k_moe_test.py
    ├── bdd100k_router_robustify_train.py
    └── bdd100k_router_robustify_test.py
```

- `bdd100k_router_robustify_train.py` trains the robust router/sentinel from a
  baseline checkpoint.
- `bdd100k_router_robustify_test.py` evaluates clean behavior and multiple
  attacks, including PGD, transfer PGD, Square, NES, HopSkipJump, and Boundary.
- The baseline train and test scripts provide the imported architecture,
  dataset, and metric helpers required by the robustification scripts.


## Required baseline assets

For the initial model training, testing, and data-preprocessing implementation,
refer to the [baseline CSA-GMoE
repository](https://github.com/badjie90/CSA-GMoE-Context-Aware-Multi-Label-Object-Presence-Prediction-in-Autonomous-Driving.git).
Those baseline scripts are essential dependencies of this project: they define
the model architecture, prepare the reproducible metadata splits, train the
original MoE checkpoint, and provide evaluation utilities imported by the
robustification scripts. Use both repositories to obtain the complete pipeline
from data preparation and baseline training through robustification and robust
evaluation.

## Prerequisites from the baseline repository

Complete baseline metadata preparation and model training before using this
repository. Required inputs are:

```text
/path/to/baseline-assets/
├── metadata/
│   ├── train.json
│   ├── val.json
│   ├── test_fixed.json
│   └── metadata_bundle.json
└── moe_stage3/
    ├── config.json
    └── checkpoints/
        └── best.pt
```

The underlying images and annotations are available from the [official BDD100K
download page](https://bdd-data.berkeley.edu/download.html). Image paths stored
in the metadata must be valid on the robustification machine.

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended.

```bash
git clone <ROBUSTIFICATION-REPOSITORY-URL>
cd <ROBUSTIFICATION-REPOSITORY>
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Suggested `requirements.txt`:

```text
numpy>=1.26
pillow>=10.0
matplotlib>=3.8
scikit-learn>=1.4
timm>=1.0.15
torch>=2.1
torchvision>=0.16
torchmetrics>=1.3
foolbox>=3.3.4
```

Install the correct CUDA-enabled PyTorch build from the [official PyTorch
installer](https://pytorch.org/get-started/locally/) when necessary.

## Configure input paths

Run all commands from the repository root:

```bash
METADATA_DIR=/path/to/baseline-assets/metadata
BASELINE_RUN=/path/to/baseline-assets/moe_stage3
ROBUST_DIR=outputs/router_sentinel
```

Pass these paths explicitly. The original scripts contain machine-specific
absolute default paths that are not portable to a new clone.

## Run order

### 1. Verify the baseline checkpoint

Before robustification, confirm that the baseline checkpoint and fixed test set
load correctly:

```bash
python scripts/bdd100k_moe_test.py \
  --train-script scripts/bdd100k_moe_train.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$BASELINE_RUN" \
  --checkpoint-name best.pt \
  --output-dir outputs/baseline_check \
  --batch-size 64 \
  --device cuda:0
```

### 2. Train the robust router and sentinel

```bash
python scripts/bdd100k_router_robustify_train.py \
  --train-script scripts/bdd100k_moe_train.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$BASELINE_RUN" \
  --checkpoint-name best.pt \
  --output-dir "$ROBUST_DIR" \
  --max-train-samples 4000 \
  --max-val-samples 1000 \
  --batch-size 16 \
  --warmup-epochs 2 \
  --joint-epochs 50 \
  --attack-eps 0.05 \
  --attack-steps 5 \
  --adv-fraction 0.50 \
  --clean-accept-target 0.90 \
  --device cuda:0 \
  --seed 42
```

The script freezes or trains model components according to its configuration,
creates adversarial routing examples, and optimizes the router and sentinel.
Use `--train-backbone` only when you intentionally want to update the visual
backbone; this increases memory and compute requirements.

List every available optimization and attack option with:

```bash
python scripts/bdd100k_router_robustify_train.py --help
```

### 3. Evaluate robustification

The robustification evaluator can test multiple attacks in one run. Inspect the
interface first if you changed the training output layout:

```bash
python scripts/bdd100k_router_robustify_test.py --help
```

Then run an evaluation using the paths printed or saved by robust training. A
representative command structure is:

```bash
python scripts/bdd100k_router_robustify_test.py \
  --robustify-script scripts/bdd100k_router_robustify_train.py \
  --train-script scripts/bdd100k_moe_train.py \
  --test-script scripts/bdd100k_moe_test.py \
  --metadata-dir "$METADATA_DIR" \
  --robust-run-dirs "$ROBUST_DIR" \
  --checkpoint-name best.pt \
  --output-dir outputs/robust_evaluation \
  --attacks pgd transfer_pgd square nes hsj boundary \
  --max-samples 50 \
  --device cuda:0
```

The evaluator loads `best.pt` from each directory supplied through
`--robust-run-dirs`. For an initial smoke test, use one attack and
`--max-samples 10`.

## Evaluation goals

Compare the robustified and baseline systems using both clean and adversarial
measurements:

- clean fused object-presence performance;
- clean router agreement and sentinel acceptance;
- attack success and router flip rates;
- adversarial fused and expert-level performance;
- sentinel detection/rejection behavior;
- robust coverage and performance on accepted samples;
- perturbation magnitude and black-box query cost;
- runtime and inference overhead.

Report clean performance together with adversarial performance. A defense that
rejects most clean inputs or substantially reduces clean accuracy should not be
described using attack success rate alone.

## Reproducibility

Record the baseline checkpoint, robust checkpoint, fixed metadata split, seed,
attack norm and epsilon, steps or query budget, package versions, and the full
command for each result. Use the same test samples when comparing the baseline
and robustified model.

Black-box attacks are expensive. Validate the pipeline on a few samples before
launching the complete evaluation, and keep each attack in a separate output
directory when running it independently.


