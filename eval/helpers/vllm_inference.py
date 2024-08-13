import os
from vllm import LLM, SamplingParams
os.environ['HF_TOKEN'] = # TODO: <hf token here>
model_name_or_path = 'google/gemma-2-9b-it'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# TODO: set prompt !!!! # from smaug_dpo_single_ling_doc_summary.txt (Aspect based summ)
messages = [
{"role": "user", "content": prompt},
]
temp_text =  tokenizer.apply_chat_template(
messages,
tokenize = False,
add_generation_prompt = True)
llm = LLM(model=model_name_or_path, dtype='half') #, dtype='half' for Gemma - since rtx8000 has no bfloat16 - only float16
sampling_params = SamplingParams(temperature=0.0, max_tokens=1624)
outputs = llm.generate(temp_text, sampling_params)
outputs[0].outputs[0].text
