from huggingface_hub import InferenceClient
from llama_cpp import Llama
from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

# Initialize client pointing to TGI endpoint
client = InferenceClient(
    model="http://localhost:8080",  # URL to the TGI server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
    stop_sequences=[],
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Initialize client pointing to TGI endpoint
client = OpenAI(
    base_url="http://localhost:8080/v1",  # Make sure to include /v1
    api_key="not-needed",  # TGI doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Initialize client pointing to llama.cpp server
client = InferenceClient(
    model="http://localhost:8080/v1",  # URL to the llama.cpp server
    token="sk-no-key-required",  # llama.cpp server requires this placeholder
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Initialize client pointing to llama.cpp server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",  # llama.cpp server requires this placeholder
)

# Chat completion
response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",  # Model identifier can be anything as server only loads one model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Initialize client pointing to vLLM endpoint
client = InferenceClient(
    model="http://localhost:8000/v1",  # URL to the vLLM server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Initialize client pointing to vLLM endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

client = InferenceClient(model="http://localhost:8080")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Raw text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    details=True,
)
print(response.generated_text)

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# Advanced parameters example
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,  # Higher for more creativity
)
print(response.choices[0].message.content)

client = InferenceClient(model="http://localhost:8080/v1", token="sk-no-key-required")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# For direct text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    details=True,
)
print(response.generated_text)

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")

# Advanced parameters example
response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Nucleus sampling probability
    frequency_penalty=0.5,  # Reduce repetition of frequent tokens
    presence_penalty=0.5,  # Reduce repetition by penalizing tokens already present
    max_tokens=200,  # Maximum generation length
)
print(response.choices[0].message.content)
# Using llama-cpp-python package for direct model access

# Load the model
llm = Llama(
    model_path="smollm2-1.7b-instruct.Q4_K_M.gguf",
    n_ctx=4096,  # Context window size
    n_threads=8,  # CPU threads
    n_gpu_layers=0,  # GPU layers (0 = CPU only)
)

# Format prompt according to the model's expected format
prompt = """<|im_start|>system
You are a creative storyteller.
<|im_end|>
<|im_start|>user
Write a creative story
<|im_end|>
<|im_start|>assistant
"""

# Generate response with precise parameter control
output = llm(
    prompt,
    max_tokens=200,
    temperature=0.8,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["<|im_end|>"],
)

print(output["choices"][0]["text"])

client = InferenceClient(model="http://localhost:8000/v1")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# For direct text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Advanced parameters example
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=200,
)
print(response.choices[0].message.content)

# Initialize the model with advanced parameters
llm = LLM(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    gpu_memory_utilization=0.85,
    max_num_batched_tokens=8192,
    max_num_seqs=256,
    block_size=16,
)

# Configure sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    max_tokens=100,  # Maximum length
    presence_penalty=1.1,  # Reduce repetition
    frequency_penalty=1.1,  # Reduce repetition
    stop=["\n\n", "###"],  # Stop sequences
)

# Generate text
prompt = "Write a creative story"
outputs = llm.generate(prompt, sampling_params)
print(outputs[0].outputs[0].text)

# For chat-style interactions
chat_prompt = [
    {"role": "system", "content": "You are a creative storyteller."},
    {"role": "user", "content": "Write a creative story"},
]
formatted_prompt = llm.get_chat_template()(chat_prompt)  # Uses model's chat template
outputs = llm.generate(formatted_prompt, sampling_params)
print(outputs[0].outputs[0].text)
client.generate(
    "Write a creative story",
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    top_k=50,  # Consider top 50 tokens
    max_new_tokens=100,  # Maximum length
    repetition_penalty=1.1,  # Reduce repetition
)
# Via OpenAI API compatibility
response = client.completions.create(
    model="smollm2-1.7b-instruct",  # Model name (can be any string for llama.cpp server)
    prompt="Write a creative story",
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    frequency_penalty=1.1,  # Reduce repetition
    presence_penalty=0.1,  # Reduce repetition
    max_tokens=100,  # Maximum length
)

# Via llama-cpp-python direct access
output = llm(
    "Write a creative story",
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_tokens=100,
    repeat_penalty=1.1,
)
params = SamplingParams(
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Consider top 95% probability mass
    top_k=50,  # Consider top 50 tokens
    max_tokens=100,  # Maximum length
    presence_penalty=0.1,  # Reduce repetition
)
llm.generate("Write a creative story", sampling_params=params)
client.generate(
    "Write a varied text",
    repetition_penalty=1.1,  # Penalize repeated tokens
    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
)
# Via OpenAI API
response = client.completions.create(
    model="smollm2-1.7b-instruct",
    prompt="Write a varied text",
    frequency_penalty=1.1,  # Penalize frequent tokens
    presence_penalty=0.8,  # Penalize tokens already present
)

# Via direct library
output = llm(
    "Write a varied text",
    repeat_penalty=1.1,  # Penalize repeated tokens
    frequency_penalty=0.5,  # Additional frequency penalty
    presence_penalty=0.5,  # Additional presence penalty
)
params = SamplingParams(
    presence_penalty=0.1,  # Penalize token presence
    frequency_penalty=0.1,  # Penalize token frequency
)
client.generate(
    "Generate a short paragraph",
    max_new_tokens=100,
    min_new_tokens=10,
    stop_sequences=["\n\n", "###"],
)
# Via OpenAI API
response = client.completions.create(
    model="smollm2-1.7b-instruct",
    prompt="Generate a short paragraph",
    max_tokens=100,
    stop=["\n\n", "###"],
)

# Via direct library
output = llm("Generate a short paragraph", max_tokens=100, stop=["\n\n", "###"])
params = SamplingParams(
    max_tokens=100,
    min_tokens=10,
    stop=["###", "\n\n"],
    ignore_eos=False,
    skip_special_tokens=True,
)

engine_args = AsyncEngineArgs(
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    gpu_memory_utilization=0.85,
    max_num_batched_tokens=8192,
    block_size=16,
)

llm = LLM(engine_args=engine_args)
