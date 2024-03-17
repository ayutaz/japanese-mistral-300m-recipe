from transformers import MistralForCausalLM, MistralConfig
import json

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = MistralConfig.from_dict(config)
    return config

def get_model_size(model):
    model_size = sum(t.numel() for t in model.parameters())
    return model_size / 1000**2  # Convert to millions

# Load Mistral config
config = load_config_from_json(config_file="config.json")
print(config)

# Create Mistral model
model = MistralForCausalLM(config)
print(model)

# Calculate and print model size
mistral_size = get_model_size(model)
print(f"Mistral-300m size: {mistral_size:.1f}M parameters")