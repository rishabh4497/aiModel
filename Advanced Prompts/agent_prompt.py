
from llama import Llama
import fire

def role_simulator(role: str, ckpt_dir: str, tokenizer_path: str):
    # Initialize the Llama model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=192,
        max_batch_size=4,
    )

    first_question = 'What do you think is the best way to reduce poverty?'

    # Generate a response as per the role
    prompt = f"As a {role}, how would you answer the question: '{first_question}'?"
    response = generator.text_completion(
        [prompt],
        max_gen_len=100,  # Lowering this value for now, you can adjust as needed
        temperature=0.2,
        top_p=0.9,
    )[0]['generation']

    print(f"{role}: {response}")

if __name__ == "__main__":
    fire.Fire(role_simulator)
