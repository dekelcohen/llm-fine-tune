import time
import os
import re
from pathlib import Path
import ast 
from loguru import logger
import pandas as pd
from openai import AzureOpenAI, RateLimitError
from dotenv import load_dotenv

# Load environment variables
load_dotenv() 
# .env KEY=VALUE files at python interpreter cwd OR bash: export KEY=VALUE in cmdline ..
# See samples\GPT-3\LLM_code\OpenAI_api\.env
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") # 'https://<ours-open-ai-resource-name>.openai.azure.com/'
api_version = os.getenv("AZURE_OPENAI_API_VERSION") # "2023-07-01-preview"
api_key = os.getenv("AZURE_OPENAI_API_KEY")


client = AzureOpenAI(azure_endpoint=azure_endpoint,
api_version=api_version,
api_key=api_key)


def create_template_class(class_list, text, prompt = "Please classify the following text in regards to the war in Gaza strip."):       
    for i, (key, value) in enumerate(class_list.items(), start=1):  
        prompt += f"\n{i}. {value}: classify as \"{key}\""  
    prompt += f"\nText: {text}"  
    return prompt  


def save_split_df_chunks(test_df, chunk_size=300, folder_path='./temp/data_splits'):  
    folder = Path(folder_path)  
    folder.mkdir(parents=True, exist_ok=True)  

    num_chunks = (len(test_df) + chunk_size - 1) // chunk_size  

    for i in range(num_chunks):  
        chunk_df = test_df[i * chunk_size:(i + 1) * chunk_size]  
        chunk_df.to_parquet(folder / f'data_split_{i}.parquet')  

def internal_invoke_gpt(messages, llm_params):
    """
    Internal func - called by retry-wrapper invoke_gpt below
    """            
    try:
        response = client.chat.completions.create(model=llm_params['model'],
        messages = messages,
        temperature=llm_params['generation']['temperature'],
        max_tokens=llm_params['generation']['max_tokens'],
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
        return response.choices[0].message.content
    except RateLimitError as e:
        print(f"RateLimitError Ex occurred: {e}")        
        return e
    except Exception as e:
        # Handle specific retryable exceptions (e.g., TimeoutError, ServiceUnavailableError)
        # You can log the exception or take other appropriate actions here
        print(f"Exception occurred: {e}")                
        return None  # Return None to indicate that the call should be retried

def invoke_gpt(messages, llm_params):
    """
    Call GPT api up to three times. If a RateLimitError is encountered, it will sleep for the 
    specified duration and then retry. If all three attempts fail, the function will return None.

     Parameters
     ----------
     messages : OpeNAI api to prompt the llm by role and content.
     Ex:
         messages = [{"role":"system","content":"You are an AI assistant that helps translating to hebrew dataset for entity matching. the Task is to translate pairs of matching and non-matching organization names in hebrew.\nbusiness suffixes should be translated\nGmbH, LLC, AS, A.D., plc, Ltd., s r o, S.A.S., S.R.L., PLC, AG, Inc., Pte Ltd, Tbk translate to בע'מ Co. to חברה Corp. to תאגיד Holding(s) to אחזקות\nHere are some examples in english: cargill deutschland matches cargill deutschland gmbh.\ntokens_str: cargill deutschland\tm_tokens_str: cargill deutschland gmbh gt_label:1\ntokens_str: integrated biotherapeutics united states m_tokens_str: cara therapeutics united states gt_label:0\n"},
                     {"role":"user","content":"Translate: \ntokens_str: nu mega technologies inc m_tokens_str:numega technologies gt_label:1"},
                     {"role":"assistant","content":"tokens_str: נומגה טכנולוגיות m_tokens_str: נומגה טכנולוגיות בע\"מ gt_label:1\ntokens_str: נומגה טכנולוגיות סין m_tokens_str: נומגה טכנולוגיות ארה\"ב gt_label:1\n"},
                     {"role":"user","content":"tokens_str: brenco trading international\tm_tokens_str: shell trading international limited gt_label:0\ntokens_str: national athletic trainers association\tm_tokens_str: national wheelchair athletic association gt_label:0"},
                     {"role":"assistant","content":"tokens_str: ברנקו טריידינג בינלאומי  m_tokens_str: של טריידינג בינלאומי בע\"מ gt_label:0\ntokens_str: איגוד המאמנים האתלטיים הלאומי m_tokens_str: איגוד האתלטים הלאומי בכסאות גלגלים gt_label:0"},
                     {"role":"user","content":"tokens_str: bank of america m_tokens_str: the america bank gt_label: 1"}]
         engine : str, gpt model deployment string in openai azure-api "gpt-35-turbo".
         max_tokens : int, optional The default is 4000.
         temperature : model diversity - default is 0.3.
    
    Returns
    -------
    response : string of gpt response, or None if failed

    """   
    for retry_attempt in range(3):  
        response = internal_invoke_gpt(messages, llm_params)  

        if not isinstance(response, RateLimitError):  
            return response

        # Extract the number of seconds from the error message  
        match = re.search(r"(\d+) seconds", str(response))  
        if match:  
            sleep_seconds = int(match.group(1))  
        else:  
            sleep_seconds = 10  

        print(f"Sleeping for {sleep_seconds} seconds due to rate limit. Retry attempt {retry_attempt + 1}.")  
        time.sleep(sleep_seconds)  

    print("All retry attempts failed. Returning None.")  
    return None


def internal_invoke_tgi(messages, engine, max_tokens, temperature):
    from huggingface_hub import InferenceClient
    """
    Internal func - called by retry-wrapper invoke_gpt below
    """
    cont_fl=1
    try:
        client = InferenceClient(model="http://10.32.60.10:8080")

        # for token in client.text_generation("How do you make cheese?", max_new_tokens=12, stream=True):
        #     print(token)
        tgi_message = ''

        for i, val in enumerate(messages):
            tgi_message = tgi_message + val["role"] + "\n"
            tgi_message = tgi_message + val["content"] + "\n"
        if cont_fl == 0:
            output = client.text_generation(prompt=tgi_message)
        if cont_fl == 1:
            # generation parameter
            gen_kwargs = dict(
                max_new_tokens=512,
                top_k=30,
                top_p=0.9,
                temperature=0.2,
                repetition_penalty=1.02,
                stop_sequences=["\nUser:", "<|endoftext|>", "</s>","\nuser"],
            )
            output = client.text_generation(prompt=tgi_message, **gen_kwargs)

        return output
    except:
        print("All retry attempts failed. Returning None.")
        return None

def invoke_tgi(messages, engine="gpt-35-turbo-16k", max_tokens=4000, temperature=0.3):
    try:
        response = internal_invoke_tgi(messages, engine, max_tokens, temperature)
        return response
    except:
        print("All retry attempts failed. Returning None.")
        return None


def apply_gpt_to_row(row, messages_template, llm_params):  
    """  
    Calls LLM for one row of df --> return LLM response (text - not, yet parsed)  
      
    Parameters:  
    - row: A single row of the DataFrame.  
    - messages_template: List of message templates containing placeholders.  
        - each placeholder is a column name from df row (ex: <<<!col text1!>>>) to be replaced by the row['col text1'] value
    - max_input_len: Maximum length of input text for each placeholder value.  
    - delay_between_reqs: Delay between requests to the LLM.  
    - flag_LLM: Flag to determine which LLM to use ('GPT' or 'TGI').  
      
    Returns:  
    - response: The response from the LLM.  
    """  
    messages_copy = []  
      
    for message in messages_template:  
        message_copy = message.copy()  
          
        # Find all placeholders in the message content  
        placeholders = re.findall(r'<<<!(.*?)!>>>', message_copy["content"])  
          
        # Replace placeholders with actual column values  
        for placeholder in placeholders:  
            if placeholder in row:  
                value = str(row[placeholder])  
                if len(value) > llm_params['max_input_len']:
                    value = value[:llm_params['max_input_len']]  
                message_copy["content"] = message_copy["content"].replace(f'<<<!{placeholder}!>>>', value)  
            else:  
                raise KeyError(f"Placeholder column '{placeholder}' not found in row with index {row.name}")  
          
        messages_copy.append(message_copy)  
      
    if llm_params['model_type'] == 'GPT':  
        response = invoke_gpt(messages_copy, llm_params)  
        time.sleep(llm_params['delay_between_reqs'])  
    elif llm_params['model_type'] == 'TGI':  
        response = invoke_tgi(messages_copy)  
      
    logger.info(f"Processed row index: {row.name}")  
    return response  
    

def apply_gpt_df(df, messages_template, llm_params):
    """
    Apply a GPT prompt to every row in a dataframe with prompt template (messages) with the row[col_name] is the text to be inserted into the template
    Usage Example:
    # Load dataframe (assuming you already have a dataframe named 'df')  
    # df = pd.read_csv("your_data.csv")  
      
    # Define the messages template as shown in the example - see invoke_gpt docs
    messages_template = [...]  
      
    # Apply GPT to the dataframe  
    output_df = apply_gpt_df(df, 'column_name', messages_template)  
    

    Parameters
    ----------
    df : df
    col_name : column containing text to insert into the prompt
    messages_template : array of dictionaries of openai api role prompts - see invoke_gpt doc for example 

    Returns
    -------
    gpt output - pd.Series of strs with GPT response

    """    
    gpt_output = df.apply(apply_gpt_to_row, axis=1, messages_template=messages_template, llm_params=llm_params)
    return gpt_output


def apply_gpt_data_splits(messages_template, llm_params, data_splits_folder_path='./temp/data_splits', output_folder_path='./temp/data_splits_output'):
    """
    reads the parquet files from the data_splits_folder_path, applies the apply_gpt_df function to each DataFrame, 
    and saves the result in a separate folder with the same name and _output suffix:
    """
    input_folder = Path(data_splits_folder_path)  
    output_folder = Path(output_folder_path)  
    output_folder.mkdir(parents=True, exist_ok=True)  
  
    for input_file in input_folder.glob('*.parquet'):  
        output_file = output_folder / (input_file.stem + '_output.parquet')  
          
        if not output_file.exists():  
            test_df = pd.read_parquet(input_file)  
            test_df['gpt_out'] = apply_gpt_df(test_df, messages_template=messages_template, llm_params=llm_params)
            test_df.to_parquet(output_file, engine='pyarrow')  
            logger.info(f"Processed and saved {output_file}")  
        else:  
            logger.info(f"Skipped {output_file}")

def filter_bad_gpt_outputs(df, gpt_out_col_name = 'gpt_out'):
  df = df.dropna(subset=['gpt_out']) # Remove timeouts and request failure to GPT - no output    
  # Filter out rows with gpt outputs that do not match the expected json format: {"Class": ["injured", "medical", "..." , "..." ]}
  pattern = r'^\{"Class":\s*\[("[a-zA-Z-]+",\s*)*"[a-zA-Z-]+"\]\}$'    
  # Filter out rows that don't match the pattern  
  df = df[df['gpt_out'].str.match(pattern)]      
  return df

def parse_class(df, class_list, gpt_out_col_name = 'gpt_out', filter_bad_output = True):  
    """
    Parse GPT response (output) - if the output is of the shape {'Class': ['injured', 'medical', .... ]}

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    class_list : list of strings of class names ex: ['injured', 'medical', .... ]
    gpt_out_col_name: the col_name in df that contains gpt json result 
    filter_bad_output: optional filter bad gpt outputs that do not match the expected json format: {"Class": ["injured", "medical", "..." , "..." ]}
    Returns
    -------
    df : Same df with added boolean columns, one per each class 
         Ex: df.iloc[0]
             Text           {'Class': ['injured', 'medical']}
             class_injured                                     True
             class_medical                                     True
             class_water                                      False
             class_fuel                                       False
             ....
    """
    if filter_bad_output:
        df = filter_bad_gpt_outputs(df, gpt_out_col_name)
        
    df[gpt_out_col_name + '_dct'] = df[gpt_out_col_name].apply(ast.literal_eval)
    for c in class_list.keys():  
        df['class_' + c] = df[gpt_out_col_name + '_dct'].apply(lambda x: c in x['Class'])  
    return df  
  
