import subprocess
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from tqdm.auto import tqdm
import threading
import time
import urllib.parse

# Define widgets
prompt_input = widgets.Textarea(placeholder='Enter your prompt here...', layout={'width': '100%', 'height': '200px'})
generate_button = widgets.Button(description='Generate Text')
output_area = widgets.Output()

# Define button callback
def on_button_click(b):
    with output_area:
        clear_output()  # Clear previous output
        if prompt_input.value:
            # Save the user input to a file
            input_path = '{replace_input_path}'
            with open(input_path, 'w') as file:
                file.write(prompt_input.value)

            # Define the output file path
            output_path = '{replace_output_path}'

            # Run the script using torchrun
            command = f"torchrun --nproc_per_node 1 {script_path} {input_path} {output_path} CodeLlama-7b/ CodeLlama-7b/tokenizer.model"

            def run_process():
                process = subprocess.run(command, shell=True, text=True, capture_output=True)
                if process.stderr:
                    print(f"Error: {process.stderr}")
                else:
                    # Load and display the output
                    with open(output_path, 'r') as file:
                        output = file.read()
                        if output.strip():
                            print(output)
                        else:
                            # If output is blank, generate a Google search URL
                            query = urllib.parse.quote(prompt_input.value)
                            url = f'https://www.google.com/search?q={query}'
                            display(HTML(f'Results are blank. Perform a <a href="{url}" target="_blank">Google search</a>.'))

            # Start a thread to run the process
            process_thread = threading.Thread(target=run_process)
            process_thread.start()

            # Display a loading bar while the process is running
            with tqdm(total=1, desc="Generating output") as pbar:
                while process_thread.is_alive():
                    time.sleep(0.5)
                pbar.n = 1
                pbar.last_print_n = 1
                pbar.refresh()
        else:
            print('Please enter a prompt.')

# Bind callback to button
generate_button.on_click(on_button_click)

# Display widgets
display(prompt_input, generate_button, output_area)

