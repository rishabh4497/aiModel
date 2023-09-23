from llama import Llama
import sys
import json

def main(input_path, output_path):
    # Load user input
    with open(input_path, 'r') as file:
        prompts = [line.strip() for line in file]

    # Setup the model
    generator = Llama.build(
        ckpt_dir='CodeLlama-7b/',
        tokenizer_path='CodeLlama-7b/tokenizer.model',
        max_seq_len=192,
        max_batch_size=4,
    )

    # Generate text
    results = generator.text_completion(
        prompts,
        max_gen_len=None,
        temperature=0.2,
        top_p=0.9,
    )

    # Write the output to a file
    with open(output_path, 'w') as file:
        for result in results:
            file.write(result['generation'] + '\\n')

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)