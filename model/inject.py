"""
Injection logic to insert SRA modules into a pretrained transformer.

We wrap the model's layers so that after specific layer forward passes,
the hidden states are processed through an SRA module before continuing.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from .sra_module import SRAModule
from .sra_mamba import SRAMambaModule


class SRAWrapper(nn.Module):
    """
    Wraps a transformer layer + SRA module.

    Forward pass:
        1. Run transformer layer
        2. Pass output through SRA module
        3. Return modified hidden states
    """

    def __init__(self, original_layer, sra_module):
        super().__init__()
        self.original_layer = original_layer
        self.sra_module = sra_module

    def __getattr__(self, name):
        """Delegate attribute access to the original layer."""
        # First try to get from this instance
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If not found, try the original layer
            return getattr(self.original_layer, name)

    def forward(self, hidden_states, *args, **kwargs):
        # Run original layer
        outputs = self.original_layer(hidden_states, *args, **kwargs)

        # Extract hidden states (handle tuple output from some layers)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            other_outputs = outputs[1:]
        else:
            hidden_states = outputs
            other_outputs = ()

        # Apply SRA
        hidden_states = self.sra_module(hidden_states)

        # Return in same format as original layer
        if other_outputs:
            return (hidden_states,) + other_outputs
        return hidden_states


def inject_sra(model, config):
    """
    Inject SRA modules into a pretrained model.

    Args:
        model: Pretrained transformer model (e.g., Llama)
        config: Dict with SRA configuration:
            - d_adapter: int
            - state_dim: int
            - n_channels: int
            - delta_inits: list of float
            - gate_bias_init: float
            - inject_layers: list of int (layer indices to inject after)

    Returns:
        model: Modified model with SRA modules injected
        sra_modules: List of injected SRA modules (for tracking)
    """
    print("Injecting SRA modules...")

    # Get model config
    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    print(f"  Model: {model.config._name_or_path if hasattr(model.config, '_name_or_path') else 'unknown'}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Total layers: {n_layers}")
    print(f"  Injecting at layers: {config['inject_layers']}")

    # Validate injection points
    for layer_idx in config['inject_layers']:
        if layer_idx >= n_layers:
            raise ValueError(f"Layer {layer_idx} out of range (model has {n_layers} layers)")

    # Get the layers module (Llama uses model.model.layers)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        layers = model.transformer.h
    else:
        raise ValueError("Could not find transformer layers in model")

    # Create and inject SRA modules
    use_mamba = config.get('use_mamba', False)
    sra_modules = []
    for layer_idx in config['inject_layers']:
        # Create SRA module
        if use_mamba:
            sra = SRAMambaModule(
                hidden_dim=hidden_dim,
                d_adapter=config.get('d_adapter', 128),
                d_state=config.get('d_state', 16),
                d_conv=config.get('d_conv', 4),
                expand_factor=config.get('expand_factor', 2),
                gate_bias_init=config.get('gate_bias_init', 0.0),
            )
        else:
            sra = SRAModule(
                hidden_dim=hidden_dim,
                d_adapter=config['d_adapter'],
                state_dim=config['state_dim'],
                n_channels=config['n_channels'],
                delta_inits=config['delta_inits'],
                gate_bias_init=config['gate_bias_init'],
            )

        # Move SRA to same device as model
        sra = sra.to(model.device)

        # Wrap the layer
        original_layer = layers[layer_idx]
        wrapped_layer = SRAWrapper(original_layer, sra)
        layers[layer_idx] = wrapped_layer

        sra_modules.append(sra)
        print(f"  ✓ Injected SRA at layer {layer_idx}")

    # Freeze base model parameters
    print("\nFreezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze SRA parameters
    for sra in sra_modules:
        for param in sra.parameters():
            param.requires_grad = True

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter summary:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.3f}%")

    return model, sra_modules


def load_model_with_sra(model_name, sra_config, device='cuda', torch_dtype=torch.bfloat16):
    """
    Load a pretrained model and inject SRA modules.

    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.2-1B")
        sra_config: SRA configuration dict
        device: Device to load model on
        torch_dtype: Data type for model

    Returns:
        model: Model with SRA injected
        tokenizer: Tokenizer
        sra_modules: List of SRA modules
    """
    from transformers import AutoTokenizer

    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    # Inject SRA
    model, sra_modules = inject_sra(model, sra_config)

    return model, tokenizer, sra_modules


if __name__ == "__main__":
    # Test injection (requires model download)
    import sys

    print("Testing SRA injection...")
    print("Note: This will download Llama-3.2-1B (~5GB) if not cached\n")

    # Ask user to confirm
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Skipping test.")
        sys.exit(0)

    # SRA config
    sra_config = {
        'd_adapter': 128,
        'state_dim': 64,
        'n_channels': 3,
        'delta_inits': [0.001, 0.01, 0.1],
        'gate_bias_init': -5.0,
        'inject_layers': [4, 8, 12],
    }

    # Load model
    model, tokenizer, sra_modules = load_model_with_sra(
        "meta-llama/Llama-3.2-1B",
        sra_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        torch_dtype=torch.bfloat16,
    )

    # Test generation
    print("\nTesting generation...")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")

    print("\n✅ Injection successful! Model can generate text.")
