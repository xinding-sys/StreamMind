import os
import ast
import json
import time
import argparse
import traceback
from tqdm import tqdm
from multiprocessing.pool import Pool

from openai import AzureOpenAI



from azure.ai.inference import ChatCompletionsClient
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential
import os
import re




def init():
    credential = ChainedTokenCredential(
    AzureCliCredential(),
    DefaultAzureCredential(
        exclude_cli_credential=True,
        # Exclude other credentials we are not interested in.
        exclude_environment_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_developer_cli_credential=True,
        exclude_powershell_credential=True,
        exclude_interactive_browser_credential=True,
        exclude_visual_studio_code_credentials=True,
        # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
        # Azure ML Compute jobs that has the client id of the
        # user-assigned managed identity in it.
        # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
        # In case it is not set the ManagedIdentityCredential will
        # default to using the system-assigned managed identity, if any.
        managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
        )
    )
    scopes = ["api://trapi/.default"]

    api_version = '2024-10-21'  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
    model_name = 'gpt-35-turbo'  # Ensure this is a valid model name
    model_version = '1106'  # Ensure this is a valid model version
    deployment_name = re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models
    instance = 'gcr/shared/openai' # See https://aka.ms/trapi/models for the instance name
    endpoint = f'https://trapi.research.microsoft.com/{instance}/deployments/'+deployment_name

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=credential,
        credential_scopes=scopes,
        api_version=api_version
    )

    return client


def interaction(client, message_text):
    # import pdb
    # pdb.set_trace()
    # print(message_text)
    completion = client.complete(
        model='gpt-35-turbo',
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return completion


def prompt_gpt(question, answer, pred, key, qa_set, output_dir):
    # print(64644646465)
    message = [
        {
            "role": "system",
            "content":
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
        },
        {
            "role": "user",
            "content":
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
        }
    ]
    completion = interaction(client, message)
    # Convert response to a Python dictionary.
    response_message = completion.choices[0].message.content
    response_dict = ast.literal_eval(response_message)
    result_qa_pair = [response_dict, qa_set]
    # # Save the question-answer pairs to a json file.
    with open(f"{output_dir}/{key}.json", "w") as f:
        json.dump(result_qa_pair, f)


def annotate(prediction_set, caption_files, output_dir, args):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    # import pdb
    # pdb.set_trace()
    # print(prediction_set, caption_files, output_dir, args,456464)
    for file in tqdm(caption_files):
        key = file[:-5] # Strip file extension
        if key not in prediction_set:
            continue
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['p']
        try:
            prompt_gpt(question, answer, pred, key, qa_set, output_dir)
        except Exception as e:
            traceback.print_exc()
            prompt_gpt(question, answer, pred[:10], key, qa_set, output_dir)

    time.sleep(1)


def main(args):





    file = open(args.pred_path)
    pred_contents = [json.loads(line) for line in file]

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        # import pdb; pdb.set_trace()
        video_id = sample['id']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['id'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets

    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "p": pred}
        prediction_set[id] = qa_set

    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir, args) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            # import pdb
            # pdb.set_trace()
            with Pool(processes=1) as pool:
                pool.starmap(annotate, task_args)
            # for args in task_args:
            #     annotate(*args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in tqdm(os.listdir(output_dir)):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                try:
                    content = json.load(json_file)
                except:
                    print(json_file)
                    exit(0)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in tqdm(combined_contents.items()):
        try:
            # Computing score
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score

            # Computing accuracy
            pred = result[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
        except:
            print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred-path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output-dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output-json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--num-tasks", required=True, type=int, help="Number of splits.")
    parser.add_argument("--api-key", required=True, type=str, help="Azure Openai API key.")
    parser.add_argument("--api-endpoint", required=True, type=str, help="Azure Openai API endpoint.")
    parser.add_argument("--api-deployname", required=True, type=str, help="Azure Openai API deployname.")
    args = parser.parse_args()

    # Set the OpenAI API key.
    os.environ["AZURE_OPENAI_KEY"] = args.api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = args.api_endpoint
    os.environ["AZURE_OPENAI_DEPLOYNAME"] = args.api_deployname

    client = init()

    main(args)
