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

## Step-by-Step Tutorial

### Step 1: Understand the Code Structure

The demo is organized into 9 main sections:
- Installation & Setup - Import libraries and check GPU
- Model Loading - Load Llama-3.1-8B with 4-bit quantization
- LoRA Configuration - Prepare model for efficient fine-tuning
- Dataset Preparation - Format training data for chat template
- Training Configuration - Set hyperparameters and training args
- Training - Execute the fine-tuning process
- Testing - Evaluate the fine-tuned model
- Saving - Export model for deployment
- Inference - Load and use the saved model

### Step 2: Customize Your Dataset
Replace the sample `customer_support_data` with your own data:

```python
your_data = [
    {
        "instruction": "Your question or prompt",
        "input": "Optional additional context",
        "output": "Expected response from the model"
    },
    # Add more examples...
]
```

**Dataset Tips:**
- **Quality > Quantity**: 50-200 high-quality examples often work better than thousands of poor ones
- **Consistent Format**: Keep your responses in a consistent style and format
- **Diverse Examples**: Cover various scenarios your model might encounter
- **Clear Instructions**: Make sure questions are clear and unambiguous

### Step 3: Adjust Hyperparameters (Training Parameters)
Key parameters to modify based on your needs:

```python
# Model selection (choose based on your VRAM)
model_name = "unsloth/llama-3.1-8b-instruct-bnb-4bit"  # ~3GB VRAM
# model_name = "unsloth/llama-3.2-3b-instruct-bnb-4bit"  # ~2GB VRAM

# Training duration
max_steps = 60  # Quick test (increase to 200-500 for real training)
# num_train_epochs = 1  # Alternative: train for full epochs

# Learning rate (decrease if loss is unstable)
learning_rate = 2e-4  # Try 1e-4 or 5e-5 for more conservative training

# Batch size (increase if you have more VRAM)
per_device_train_batch_size = 2  # Increase to 4 or 8 with more memory
gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
```


### Step 4: Run the Training Scrip & Monitor

Watch these key metrics during training:

- Loss: Should gradually decrease from ~2.0 to ~0.5-1.0
- Memory Usage: Should stay within your GPU limits
- Training Speed: ~1-2 steps per second is normal

**Warning Signs**:

- Loss stuck at high values (>2.0) → Lower learning rate
- Loss drops to 0 → Reduce learning rate or increase dataset size
- Out of memory errors → Reduce batch size or sequence length

### Step 5: Test the Fine-tuned Model

Use the built-in testing functions:

```python
# Single prompt test
response = test_model("Your test question here")
print(response)

# Interactive chat
interactive_chat()  # Type 'quit' to exit

# Evaluate on test set
evaluate_model_on_test_set(your_test_data)
```