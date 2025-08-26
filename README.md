# Local LLM Fine-tuning Demo with Unsloth

A simple, educational repository demonstrating how to fine-tune Large Language Models on your local machine using Unsloth.

## What We'll Build
- A custom customer support chatbot based on Llama-3.1-8B
- Memory-efficient fine-tuning using QLoRA (4-bit quantization)
- Production-ready inference pipeline
- Model saving and deployment options

##  Prerequisites

### Hardware Requirements
- **GPU:** NVIDIA GPU with 3GB+ VRAM (recommended: 8GB+)
- **RAM:** 8GB+ system RAM
- **Storage:** 10GB+ free space

### Software Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **CUDA**: 11.8+ or 12.1+
- **PyTorch**: Compatible with your CUDA version

## Local Installation

### Step 1: Set Up Environment

Create a virtual environment and activate it:

```
virtualenv .venv --python=python3.11
.venv/scripts/activate
```

### Step 2: Install CUDA (if not already installed)

Check your GPU's compute capability and CUDA version:
```
nvidia-smi
```

Look at the CUDA Version field (e.g., 12.1). Then install the matching toolkit:

```
# For CUDA 12.1
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y

# For CUDA 11.8
# conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y
```

### Step 3: Install PyTorch
Choose the command based on your CUDA version.

```
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify the installation:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Step 4: Install Unsloth and Dependencies

```
# For CUDA 12.1 and Ampere GPUs (e.g., RTX 30xx, 40xx)
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 11.8: 
pip install "unsloth[cu118-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

For more on installation options, see the [Unsloth Installation Guide](https://docs.unsloth.ai/get-started/installing-+-updating/pip-install)

Install dependencies:
```
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets pandas jupyter
```

### Step 5: Verify Installation

```
import torch
import unsloth
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Unsloth version: {unsloth.__version__}")
```