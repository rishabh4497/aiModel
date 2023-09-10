from llama import Llama
import fire

def character_maker(id: str, description: str, ckpt_dir: str, tokenizer_path: str):
    generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=512,  # Increase this value
    max_batch_size=4,
    )


    valid_weapons = ['sword', 'axe', 'mace', 'spear', 'bow', 'crossbow']

    prompt = (f'The following is a character profile for an RPG game in JSON format.\n'
              f'```json\n{{\n'
              f'    \"id\": \"{id}\",\n'
              f'    \"description\": \"{description}\",\n'
              f'    \"name\": \"{{gen \'name\'}}\",\n'
              f'    \"age\": {{gen \'age\' pattern=\'[0-9]+\' stop=\',\'}},\n'
              f'    \"armor\": \"{{#select \'armor\'}}leather{{or}}chainmail{{or}}plate{{/select}}\",\n'
              f'    \"weapon\": \"{{select \'weapon\' options=valid_weapons}}\",\n'
              f'    \"class\": \"{{gen \'class\'}}\",\n'
              f'    \"mantra\": \"{{gen \'mantra\' temperature=0.7}}\",\n'
              f'    \"strength\": {{gen \'strength\' pattern=\'[0-9]+\' stop=\',\'}},\n'
              f'    \"items\": [{{#geneach \'items\' num_iterations=5 join=\', \'}}\"{{gen \'this\' temperature=0.7}}\"{{/geneach}}]\n'
              f'}}```\n')

    character_profile = generator.text_completion(
        [prompt],
        max_gen_len=500,
        temperature=0.2,
        top_p=0.9,
    )[0]['generation']

    print(f'Generated Character Profile:\n{character_profile}')

if __name__ == '__main__':
    fire.Fire(character_maker)
