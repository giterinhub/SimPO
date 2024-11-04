from datasets import load_dataset,load_from_disk
import os
import argparse
import json
from openai import OpenAI
from transformers import AutoTokenizer
import tqdm
parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--data_dir', type=str, default="/home/thanhnx/work/SimPO/SimPO/on_policy_data_gen/data/ultrafeedback-binarized-preferences-cleaned-first_20k",
                    help='Directory containing the data')
parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback",
                    help='output_dir')
args = parser.parse_args()

print(args)
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key="0fc4ed90a01c8aeba03806782c1bf531f4c9620db73fe95e278f7c348b8f671a",
)

data_dir = args.data_dir

train_dataset= load_from_disk(data_dir)

prompts = sorted(list(set(train_dataset['prompt'])))

# conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

# sampling_params = SamplingParams(temperature=args.temperature, 
#                                  top_p=args.top_p, 
#                                  max_tokens=args.max_tokens, 
#                                  seed=args.seed,)
# outputs = llm.generate(conversations, sampling_params)
outputs = []
for prompt in tqdm.tqdm(prompts):
    response = client.chat.completions.create(
        messages = [
            {"role": "user", "content": prompt}
        ],
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seed,
    )
    outputs.append(response)

tokenizer = AutoTokenizer.from_pretrained(args.model)



# Save the outputs as a JSON file.
output_data = []
for i, output in enumerate(outputs):
    messages = [{'role': 'user', 'content': prompts[i]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    generated_text = output.choices[0].message.content
    output_data.append({
        'prompt': prompts[i],
        "format_prompt": prompt,
        'generated_text': generated_text,
    })

output_file = f'output_{args.seed}.json'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
