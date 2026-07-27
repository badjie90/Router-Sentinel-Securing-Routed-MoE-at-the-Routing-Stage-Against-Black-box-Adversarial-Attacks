# Adversarial Robustification of a nuImages MoE Router

This repository trains and evaluates a robustified router for a
condition-specialized nuImages Mixture-of-Experts (MoE) model. It starts from a
trained baseline checkpoint and uses adversarial routing examples plus a
sentinel mechanism to improve or monitor routing under attack.

The underlying baseline predicts car, pedestrian, and traffic-cone presence.
Its experts specialize in location, illumination, and ego-motion.

## Repository contents

```text
.
├── README.md
├── requirements.txt
└── scripts/
    ├── nuimages_moe_train.py
    ├── nuimages_moe_test.py
    ├── nuimages_router_robustify_train.py
    └── nuimages_router_robustify_test.py
```

- `nuimages_router_robustify_train.py` trains the robust router and sentinel
  using clean and PGD-generated routing examples.
- `nuimages_router_robustify_test.py` evaluates clean behavior and PGD,
  transfer-PGD, Square, NES, HopSkipJump, and Boundary attacks.
- The baseline train and test scripts provide architecture, data, and evaluation
  helpers imported by the robustification scripts.

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

For the black-box attack implementations and their supporting evaluation
dependencies, refer to the [attack-evaluation
repository](https://github.com/badjie90/Router-Sentinel-Securing-Routed-MoE-at-the-Routing-Stage-Against-Black-box-Adversarial-Attacks.git).
That repository provides the decision-based, score-based, and transfer-based
attack logic used to test whether robustification improves the MoE router under
adversarial conditions. Access to those scripts is important for reproducing
the complete defense-evaluation workflow, including generation of adversarial
inputs, router-attack measurements, and comparisons between the original and
robustified models. Accordingly, this robustification repository should be used
together with both the baseline repository and the attack-evaluation repository
when reproducing the full experimental pipeline.

Complete baseline metadata preparation and training first:

```text
/path/to/baseline-assets/
├── nuimages_metadata/
│   ├── train.json
│   ├── val.json
│   ├── test_fixed.json
│   └── metadata_bundle.json
└── nuimages_moe_stage3/
    ├── config.json
    └── checkpoints/
        └── best.pt
```

The images referenced by the metadata must remain accessible. Download nuImages
from the [official nuImages page](https://www.nuscenes.org/nuimages) after
registration/login. Raw dataset files must not be committed to this repository.

## Installation

Python 3.10 or newer and a CUDA-capable NVIDIA GPU are recommended:

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

Choose the correct CUDA-enabled PyTorch build using the [official PyTorch
installer](https://pytorch.org/get-started/locally/).

## Configure paths

Run all commands from the repository root:

```bash
METADATA_DIR=/path/to/baseline-assets/nuimages_metadata
BASELINE_RUN=/path/to/baseline-assets/nuimages_moe_stage3
ROBUST_DIR=outputs/router_sentinel
ROBUST_RUN=outputs/router_sentinel/nuimages_moe_stage3
```

Pass paths explicitly because the scripts retain machine-specific absolute
defaults from their development environment.

## Run order

### 1. Verify the baseline checkpoint

```bash
python scripts/nuimages_moe_test.py \
  --train-script scripts/nuimages_moe_train.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$BASELINE_RUN" \
  --checkpoint-name best.pt \
  --output-dir outputs/baseline_check \
  --batch-size 64 \
  --device cuda:0
```

Do not continue until the baseline metrics and image loading are correct.

### 2. Train the robust router and sentinel

```bash
python scripts/nuimages_router_robustify_train.py \
  --train-script scripts/nuimages_moe_train.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$BASELINE_RUN" \
  --checkpoint-name best.pt \
  --output-dir "$ROBUST_DIR" \
  --max-train-samples 4000 \
  --max-val-samples 1000 \
  --val-ratio 0.15 \
  --batch-size 16 \
  --warmup-epochs 2 \
  --joint-epochs 50 \
  --lr 1e-4 \
  --attack-eps 0.05 \
  --attack-steps 5 \
  --adv-fraction 0.50 \
  --clean-accept-target 0.90 \
  --device cuda:0 \
  --seed 42
```

The default strategy does not train the backbone. Add `--train-backbone` only
when deliberately fine-tuning it, as doing so increases memory and compute
requirements. List all learning-rate, sentinel, PGD, and sampling controls with:

```bash
python scripts/nuimages_router_robustify_train.py --help
```

Training creates a child directory named after the baseline run. With the paths
above, its checkpoints are written under
`outputs/router_sentinel/nuimages_moe_stage3/checkpoints/`. Set `ROBUST_RUN` to
that child directory if your baseline run has a different name.

### 3. Evaluate the robustified router

Run an inexpensive smoke test first:

```bash
python scripts/nuimages_router_robustify_test.py \
  --robustify-script scripts/nuimages_router_robustify_train.py \
  --train-script scripts/nuimages_moe_train.py \
  --test-script scripts/nuimages_moe_test.py \
  --metadata-dir "$METADATA_DIR" \
  --robust-run-dirs "$ROBUST_RUN" \
  --checkpoint-name best.pt \
  --output-dir outputs/robust_smoke_test \
  --attacks pgd \
  --max-samples 10 \
  --batch-size 8 \
  --device cuda:0
```

Then evaluate all implemented attacks:

```bash
python scripts/nuimages_router_robustify_test.py \
  --robustify-script scripts/nuimages_router_robustify_train.py \
  --train-script scripts/nuimages_moe_train.py \
  --test-script scripts/nuimages_moe_test.py \
  --metadata-dir "$METADATA_DIR" \
  --robust-run-dirs "$ROBUST_RUN" \
  --checkpoint-name best.pt \
  --output-dir outputs/robust_evaluation \
  --attacks pgd transfer_pgd square nes hsj boundary \
  --max-samples 50 \
  --eps 0.05 \
  --queries 2000 \
  --device cuda:0
```

The evaluator loads `best.pt` from every directory supplied through
`--robust-run-dirs`. Run `python scripts/nuimages_router_robustify_test.py
--help` for PGD restarts and step size, score-based attack parameters,
HopSkipJump constraints, decision initialization, and surrogate settings.

## What to report

Compare the baseline and robustified systems using:

- clean fused object-presence metrics;
- clean router agreement and sentinel acceptance;
- attack success and router-flip rates;
- adversarial fused and expert-level performance;
- sentinel detection or rejection performance;
- accuracy and coverage on accepted samples;
- perturbation magnitude, query cost, and runtime;
- clean-performance and inference-overhead changes introduced by the defense.

A defense should not be assessed from attack success rate alone. Report whether
it rejects clean examples or reduces clean task performance.

## Reproducibility and practical guidance

Record the baseline and robust checkpoints, fixed metadata, sample subset, seed,
package versions, attack norm and epsilon, step/query budget, and full command.
Use identical test samples for baseline-versus-defense comparisons.

Black-box attacks are expensive. Test each one on a few images, reduce batch size
if GPU memory is insufficient, and save separate output directories for distinct
attack settings.

Do not commit raw nuImages files, virtual environments, caches, generated
evaluation directories, or large model checkpoints to ordinary Git history.
Use Git LFS or an artifact/release service for weights when permitted.
