# AUV-Swarm-RFL

**AUV-Swarm-RFL** is a specialized simulation framework that combines **Federated Learning (FL)** within a swarm of **Autonomous Underwater Vehicles (AUVs)**, optimized by **Deep Reinforcement Learning (DRL)** algorithms (like PPO, DDPG). 

This repository heavily focuses on dynamic resource allocation (power, frequency, and device selection) in a physically accurate underwater acoustic communication environment.

## 📁 Repository Structure
* **`config/`**: System configurations for Physics (Acoustic, Energy, Latency), FL components, and RL agents.
* **`env/`**: Core Simulation Environment (Gymnasium-based). Includes `auv_env.py` and physical models (`communication.py`, `energy.py`, `latency.py`, `reward.py`).
* **`fl_core/`**: Federated Learning framework (Simulated workers, models, datasets, aggregator, and early stopping).
* **`rl_agent/`**: Deep Reinforcement Learning implementations (PPO algorithms using Stable Baselines3).
* **`scripts/`**: Essential scripts for executing training pipelines, evaluations, and plotting charts (Fig 1 - Fig 7).

---

## 🛠 Installation

Requirements: Python 3.10+ is recommended. 
It is highly recommended to use a Virtual Environment.

```cmd
# Create virtual environment
python -m venv .venv

# Activate virual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run the Full Two-Stage Evaluation Pipeline
We have provided a comprehensive PowerShell script that executes the Beta Sensitivity analysis, Baseline training (Medium vs Full), Scheme comparison, and generates all relevant figures automatically:

```powershell
.\run_baselines_two_stage.ps1
```

### 2. Run Individual Scripts Manually
Alternatively, you can run individual processes directly from the Python CLI.

**Run FL Training manually:**
```bash
python scripts/train_baselines.py --episodes 10 --m 9 --max-fl-rounds 100 --algorithms ppo --out-dir results/my_test
```

**Evaluate Scheme Comparisons:**
```bash
python scripts/run_fig_4_5_6.py --rounds 1000 --out-dir results/eval_schemes --model-path results/fig_7_full/ppo_baseline_model
```

---

## ☁️ Google Colab Deployment

To run this repository on Google Colab (benefiting from free GPU/TPU acceleration):
1. Upload the project to your GitHub.
2. Open the file `colab_experiment.ipynb` in Google Colab.
3. Follow the cells in the notebook to interactively clone your repository, install dependencies, and run large-scale training pipelines.
