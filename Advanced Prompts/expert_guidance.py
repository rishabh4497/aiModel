
from llama import Llama
import fire

def expert_guidance(query: str, ckpt_dir: str, tokenizer_path: str):
    # Initialize the Llama model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=192,
        max_batch_size=4,
    )

    # Generate names of experts for the query
    expert_names_prompt = f"Name 3 world-class experts (past or present) who would be great at answering the question: {query}"
    expert_names = generator.text_completion(
        [expert_names_prompt],
        max_gen_len=50,
        temperature=0.2,
        top_p=0.9,
    )[0]['generation']

    print(f"Experts: {expert_names}")

    # Generate a collaborative answer from these experts
    answer_prompt = f"Please provide a joint anonymous answer from these experts to the question: {query}"
    answer = generator.text_completion(
        [answer_prompt],
        max_gen_len=200,
        temperature=0.2,
        top_p=0.9,
    )[0]['generation']

    print(f"Answer: {answer}")

if __name__ == "__main__":
    fire.Fire(expert_guidance)

