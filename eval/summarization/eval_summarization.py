# Add to PYTHONPATH D:\NLP\llm-fine-tune\eval - for from helpers.invoke_gpt
import os  
import pandas as pd  
import json  
from pathlib import Path  
from loguru import logger  
from helpers.invoke_gpt import save_split_df_chunks, apply_gpt_data_splits  
from eng_utils.common.common_util import read_df_folder  
  
def main(df, prompt_template, data_splits_folder_path='./temp/data_splits', output_folder_path='./temp/data_splits_output', max_input_len=700, flag_LLM='GPT'):  
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
    apply_gpt_data_splits(data_splits_folder_path=data_splits_folder_path, output_folder_path=output_folder_path, messages_template=messages_template, max_input_len=max_input_len, flag_LLM=flag_LLM)  
      
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
      
    # Save the output DataFrame  
    output_file = os.path.join(output_folder_path, 'evaluation_results.parquet')  
    df_out.to_parquet(output_file, engine='pyarrow')  
    logger.info(f"Evaluation results saved to {output_file}")  
  
if __name__ == "__main__":  
    # Read the input DataFrame from a Parquet file  
    input_parquet_file = './data/inputs/summarization_input.parquet'  
    df = pd.read_parquet(input_parquet_file)  
      
    # Define the prompt template  
    prompt_template = """  
    <<<!inputs!>>>  
    <<<!llm_resp_inputs!>>>  
    Task: You act as an LLM Judge:  
    Does the Summary expresses important and accurate facts from the prompt?  
    For each sentence, fact or claim stated in the Summary (Summary Unit), thoroughly check if it was correctly taken from the Text to Summarize.  
    Output Format:  
    1) Top Level Json with key "summary_factuality" : Array of Summary Unit Json objects  
    2) For each Summary Unit a Json: { "Summary" : "<summary fact/claim/sentence>", "Text" : "<Supporting text from source doc>", "Evaluation" : "<Explanation of why the Summary Unit is correct, almost all correct, partially correct or incorrect>" "Score" : "-1:incorrect|0: partially-correct|1:mostly correct|2:correct", "Aspect" : "<The relevant aspect from the prompt, if exist>" }  
    """  
      
    # Call the main function  
    main(df=df, prompt_template=prompt_template)  
