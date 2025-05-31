import os
import psutil
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.quantization import quantize_dynamic

def load_model_and_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def quantize_model(model):
    try:
        quantized_model = quantize_dynamic(
            model.to("cpu"),
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        print(f"Error during quantization: {e}")
        return None

def measure_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)
    return memory_usage_mb

def test_inference(model, tokenizer, sample_prompt):
    inputs = tokenizer(sample_prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return outputs

def main():
    model_name = "t5-small"
    model, tokenizer = load_model_and_tokenizer(model_name)

    if model is not None:
        print("Original model size:")
        original_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)  # Approximate size in MB
        print(f'Model size: {original_size:.3f} MB')

        quantized_model = quantize_model(model)

        if quantized_model is not None:
            print("Quantized model size:")
            quantized_size = sum(p.numel() for p in quantized_model.parameters()) * 4 / (1024 ** 2)  # Approximate size in MB
            print(f'Quantized model size: {quantized_size:.3f} MB')

            sample_prompt = "This is a test prompt for quantized model inference."
            outputs = test_inference(quantized_model, tokenizer, sample_prompt)
            print("Inference completed.")

if __name__ == "__main__":
    main()