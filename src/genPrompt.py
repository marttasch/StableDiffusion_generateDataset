import json
import sys
import os
from itertools import product

def generate_prompts_from_json(json_data, type="txt2img"):
    prompts = []
    
    print(f"seeds_per_prompt: {json_data[type]['seeds_per_prompt']}")
    for prompt_template in json_data[type]["prompts"]:
        print(f"prompt_template: {prompt_template}")

        # Determine which variables are used in the prompt template
        variables = ['object_names', 'object_materials', 'backgrounds', 'perspective', 'viewpoints']
        used_variables = [var for var in variables if '{' + var[:-1] + '}' in prompt_template]
        print(f"used_variables: {used_variables}")

        for combination in product(*[json_data[type][var] for var in used_variables]):
            print(f"combination: {combination}")

            # Map the variables to their values
            variable_values = dict(zip([var[:-1] for var in used_variables], combination))

            # Format the prompt
            prompt = prompt_template.format(**variable_values)

            for _ in range(json_data[type]["seeds_per_prompt"]):
                # Append the prompt to the list
                prompts.append({
                    "prompt": prompt,
                    "prompt_template": prompt_template,
                    **variable_values
                })
    return prompts

def genPrompts(datasetName=None, configFile='config.json', saveAsFile=False):
    # read config file
    with open(configFile, 'r') as jsonFile:
        promptConfig = json.load(jsonFile)

    # extract promptConfig
    promptsetsName = promptConfig['name']
    if datasetName:
        promptsetsName = datasetName

    # get prompts
    promptSets = generate_prompts_from_json(promptConfig, type="txt2img")
    print(f"Generated {len(promptSets)} prompts")

    if saveAsFile:
        # save prompts to file
        promptFileJson = os.path.join('prompts', f'{promptsetsName}.json')
        promptFileTxt = os.path.join('prompts', f'{promptsetsName}.txt')
        if not os.path.exists('prompts'):
            os.makedirs('prompts')
        with open(promptFileJson, 'w') as jsonFile:
            json.dump(promptSets, jsonFile, indent=4)
        with open(promptFileTxt, 'w') as txtFile:
            for prompt in promptSets:
                txtFile.write(prompt['prompt'] + '\n')
            print(f"Prompts saved to {promptFileJson}")
            print(f"Prompts saved to {promptFileTxt}")
    return promptSets


if __name__ == '__main__':
    print("\n### Generate prompts ###")
    # get arguments from command line
    if len(sys.argv) < 2:
        print("Missing argument: configFile")
        print("Usage: `python genPrompt.py <configFile>`")
        print("Will use default `config.json`\n")
        configFile = 'config.json'
    else:
        configFile = sys.argv[1]
        print("Using config file: ", configFile) 

    promptList = genPrompts(configFile=configFile, saveAsFile=True)