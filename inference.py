import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

# Set environment variables before importing any modules
os.environ["CUDA_VISIBLE_DEVICES"] = "5"   
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

# Define paths
BASE_MODEL_PATH = "meta-llama/Llama-2-7b-hf"
FINETUNED_MODEL_PATH = "./results/llama2-finetuned"

# Replace with your Hugging Face access token
ACCESS_TOKEN = "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"  # Replace with your actual token

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    use_auth_token=ACCESS_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

# Load the base model with quantization
print("Loading base model with 4-bit quantization...")
base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},  # Explicitly set to the first (and only) visible GPU
    torch_dtype=torch.float16,
    use_auth_token=ACCESS_TOKEN,
)

# Prepare the model for inference
print("Preparing model for inference...")
base_model = PeftModel.from_pretrained(
    base_model,
    FINETUNED_MODEL_PATH,
)
base_model.eval()

# Move model to GPU if available
if torch.cuda.is_available():
    base_model = base_model.to("cuda")
    print("Model moved to GPU.")
else:
    print("GPU not available, using CPU.")

# Define a custom stopping criterion
class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_sequence, tokenizer):
        super().__init__()
        self.stop_sequence = stop_sequence
        self.tokenizer = tokenizer
        self.stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
    
    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) < len(self.stop_ids):
            return False
        # Check if the last tokens match the stop sequence
        if input_ids[0][-len(self.stop_ids):].tolist() == self.stop_ids:
            return True
        return False

# Prepare the prompt
prompt = """Maximal translation can be defined as the process of converting a regular expression (regex) into semantically equivalent SQL LIKE operators to the fullest extent possible. If a regex cannot be fully translated into LIKE, the translation should handle as much of the regex as possible using LIKE, while the remaining portion is expressed using a simplified regex with REGEXP_LIKE. The following rules apply:

The ^ symbol at the start of a regex indicates the start of the string, and $ at the end indicates the end of the string.
If _ or % appears in the regex pattern, treat them as literals, not wildcards. Escape these characters with a backslash in the SQL LIKE clause. If * appears in regex and the character before is anything other than . then you cannot translate it. .* can be replaced by % and just . if it appears isolated as a wildcard in regex pattern it is replaced as _ .
Use parentheses in the resulting SQL query to ensure correctness and logical grouping.
For example:

Input Regex: (abc)|(def)*
Output SQL: (COLUMN LIKE '%abc%') OR REGEXP_LIKE(COLUMN, '(def)*')

Input Regex: ^ab$|cd*
Output SQL: (COLUMN LIKE 'ab') OR REGEXP_LIKE(COLUMN, 'cd*')

Input Regex: ^ab.$|cd
Output SQL: (COLUMN LIKE 'ab_') OR (COLUMN LIKE '%cd%')

Input Regex: ^ab[xy]cd$
Output SQL: (COLUMN LIKE 'abxcd' OR COLUMN LIKE 'abycd')

Input Regex: ab[^xy]cd
Output SQL: ((COLUMN LIKE '%ab_cd%') AND COLUMN NOT LIKE '%abxcd%' AND COLUMN NOT LIKE '%abycd%')

Input Regex: abc?
Output SQL: (COLUMN LIKE '%ab%' OR COLUMN LIKE '%abc%')

Input Regex: a.b_c%d
Output SQL: (COLUMN LIKE '%a_b\_c\%d%' ESCAPE '\')

Input Regex: \[\/?(embed|slider|---).*\]
Output SQL: (COLUMN LIKE '%[embed%]%' OR COLUMN LIKE '%[slider%]%' OR COLUMN LIKE '%[---%]%' OR COLUMN LIKE '%[/embed%]%' OR COLUMN LIKE '%[/slider%]%' OR COLUMN LIKE '%[/---%]%')

Given the regex, maximally rewrite it into an Oracle SQL LIKE-based equivalent wherever possible. If full translation is not achievable, ensure the output retains both LIKE and REGEXP_LIKE components for semantic accuracy.

Input regex: reg(ular expression(s|)|ex(p|es|))
### Response:
"""

# Tokenize the input
print("Tokenizing input...")
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)

if torch.cuda.is_available():
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

# Initialize stopping criteria
stopping_criteria = StoppingCriteriaList([StopOnSequence("\n###", tokenizer)])

# Generate output
print("Generating output...")
with torch.no_grad():
    output = base_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
    )

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Extract the model's response after the prompt
response_start = generated_text.find("### Response:")
if response_start != -1:
    # Extract everything after '### Response:'
    response = generated_text[response_start + len("### Response:"):].strip()
else:
    # If '### Response:' not found, attempt to extract the first 'To the following SQL query:' line
    sql_query_start = generated_text.find("To the following SQL query:")
    if sql_query_start != -1:
        # Extract everything after 'To the following SQL query:'
        response = generated_text[sql_query_start + len("To the following SQL query:"):].strip()
    else:
        # If neither is found, return the full generated text
        response = generated_text.strip()

# Further process the response to capture only the first SQL query
# Split the response by newlines and take the first relevant line
import re

# Use regex to find the first SQL query within parentheses
match = re.search(r'\(COLUMN LIKE .*?\)', response)
if match:
    response = match.group(0)

print("\nModel Output:")
print(response)
