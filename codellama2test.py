from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/youtube

!git clone https://github.com/facebookresearch/codellama.git

# Commented out IPython magic to ensure Python compatibility.
# %cd codellama

!python setup.py install

!bash download.sh

!pip install torch torchvision fire
!pip install fairscale
!pip install sentencepiece

!pip install fairscale

!ls

!fuser -k 29500/tcp  # Be careful with this command

testscript = """ from typing import Optional
from llama import Llama
import fire

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 192,
    max_gen_len: Optional[int] = None,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Sample code prompts for completion
    prompts = [
        "def add(a, b):",
        "class Person:"
    ]

    # Generate completions for the prompts
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Display the results
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)

"""

path = '/content/drive/MyDrive/youtube/codellama/test.py'

# Save the script content to the file
with open(path, 'w') as f:
    f.write(testscript)

!torchrun --nproc_per_node 1 test.py \
    --ckpt_dir CodeLlama-7b/ \
    --tokenizer_path CodeLlama-7b/tokenizer.model \
    --max_seq_len 192 --max_batch_size 4
