# cd /d D:\NLP\llm-fine-tune\eval

# TODO: Predictions with old + new finetuned unsloth on fa summary - eval both and analyze improvements 

# TODO:HIGH: chunks: 'max_tokens' - depends on the summary length (or we can split it)
  # TODO: Chunk both input doc text (that can be too long for 8K models) + response units (5 at a time)
  # TODO: large chunks with overlap - so model can see 
    # e5 emb search - not to miss relevant info
  # TODO: calculate_factuality_score - per unit, the max of all chunks 

import os  
import pandas as pd  
import json  
from pathlib import Path  
from loguru import logger  
from helpers.invoke_gpt import save_split_df_chunks, apply_gpt_data_splits  
from eng_utils.common.common_util import read_df_folder  

LLM_JUDGE_PROMPT = """
Task: You act as an LLM Judge:

Does the LLM Response expresses important and accurate facts from the prompt ?
For each sentence, fact or claim stated in the Response (called Unit), throughly check if it was correctly taken from the input Text.
Output Format: 
1) Top Level Json with key "factuality_units" : [Array of Unit Json objects]
2) For each Unit a Json: { "Response Unit" : "<response fact/claim/sentence>", "Text" : "<Supporting text from source doc>", "Evaluation" : "<Explanation of why the Response Unit is correct, almost all correct, partially correct or incorrect>" "Score" : "-1:incorrect|0: partially-correct|1:mostly correct|2:correct", "Aspect" : "<The relevant aspect from the prompt, if exist>" } 
"""  

# Function to calculate the score
def calculate_factuality_score(json_data):
    units = json_data['factuality_units']
    unit_count = len(units)
    score_sum = sum(unit['Score'] for unit in units)
    score_mean = score_sum / unit_count if unit_count else 0
    return {
        'unit_count': unit_count,
        'score_sum': score_sum,
        'score_mean': score_mean
    }




def main(df, prompt_template, llm_params, data_splits_folder_path='./temp/data_splits', output_folder_path='./temp/data_splits_output'):  
    """  
    Main function to evaluate summarization using GPT.  
      
    Parameters:  
    - df: DataFrame containing the data to be processed.  
    - prompt_template: The prompt template to be used for GPT evaluation.  
    - data_splits_folder_path: Folder path to save data splits.  
    - output_folder_path: Folder path to save GPT outputs.  
    - max_input_len: Maximum length of input text for each placeholder value.  
    - flag_LLM: Flag to determine which LLM to use ('GPT' or 'TGI').  
    """  
    # Ensure required columns are present  
    if 'inputs' not in df.columns or 'llm_resp_inputs' not in df.columns:  
        raise ValueError("Input DataFrame must contain 'inputs' and 'llm_resp_inputs' columns.")  
      
    # Save DataFrame chunks  
    save_split_df_chunks(df, folder_path=data_splits_folder_path)  
      
    # Define the messages template  
    messages_template = [  
        {"role": "system", "content": prompt_template}  
    ]  
      
    # Apply GPT to data splits  
    apply_gpt_data_splits(messages_template=messages_template, llm_params=llm_params, data_splits_folder_path=data_splits_folder_path, output_folder_path=output_folder_path)
      
    # Read the resulting folder of splits into a new DataFrame  
    df_out = read_df_folder(output_folder_path)  
      
    # Parse the 'gpt_out' column to JSON objects  
    def parse_json_safe(x):  
        try:  
            return json.loads(x) if pd.notnull(x) else None  
        except json.JSONDecodeError as e:  
            logger.error(f"JSONDecodeError for row: {x} with error: {e}")  
            return None  
      
    df_out['gpt_out_json'] = df_out['gpt_out'].apply(parse_json_safe)  

    # Apply the function to calculate the score for each row
    df_out['factuality_score'] = df_out['gpt_out_json'].apply(calculate_factuality_score)

      
    # Save the eval results output DataFrame  
    output_dir = Path('eval_results')
    # Create the directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'evaluation_results.parquet'
    df_out.to_parquet(output_file)  
    logger.info(f"Evaluation results saved to {output_file}")  
    return df_out
  
if __name__ == "__main__":  
    # Read the input containing prompts and responses
    prompts_responses_file = './summarization/test/summarization_input.parquet'  
    df = pd.read_parquet(prompts_responses_file)  
      
    # Define the prompt template for LLM response verification
    prompt_template = f"""  
    <<<!inputs!>>>  
    <<<!llm_resp_inputs!>>> 
    {LLM_JUDGE_PROMPT}
    """  
      
    # Call the main function
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") # model deployment name as created in AzureOpenAI Studio | Deployments (gpt-3.5, my-gpt-4, gpt-4o ...)
    llm_params = {
        'model_type' : 'GPT', # TGI
        'model' : deployment_name,
        'delay_between_reqs' : 1,
        'max_input_len' : 1000000, # TODO: Chunks - currently do not truncate input len
        'generation' : {
            'max_tokens' : 2500, # TODO: 'max_tokens' - depends on the summary length (or we can split it)
            'temperature' : 0.0,
        }
    }
    df_out = main(df=df, prompt_template=prompt_template, llm_params=llm_params)  
    ### *** Note: It does nothing if the output spilts already exist in temp - delete temp
    # df_out.iloc[0]['gpt_out_json']
    df_out.iloc[0]['factuality_score']
