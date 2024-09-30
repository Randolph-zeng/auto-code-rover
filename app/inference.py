import inspect
import json
import re
from collections.abc import Callable
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

from loguru import logger
from termcolor import colored
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from app import globals
from app.api.manage import ProjectApiManager
from app.data_structures import FunctionCallIntent, MessageThread
from app.log import (
    log_and_cprint,
    log_and_print,
    print_acr,
    print_banner,
    print_issue,
    print_retrieval,
)
from app.model import common, ollama
from app.search.search_manage import SearchManager
from app.utils import parse_function_invocation, parse_git_patch
from app.api.agent_proxy import is_valid_response
from app.post_process import extract_diff_one_instance, ExtractStatus



# FIXME: the system prompt should be different for stratified/state machine.
SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Your task is to invoke a few search API calls to gather buggy information, then write patches to solve the issues.
"""

# FIXME: Adjust the response according to the parsing rule of search actions 
DEFAULT_CRITIC_RESPONSE = """Failed to parse the search actions or bug locations out of the response. The result generated might not be following the instructions given. 
Please check your response and follow the **exact** format desired strictly.    
"""


def prepare_explanation_prompt(repo, issue_description, ground_truth_patch, relevant_context):
    explanation_prompt = f"""Hi, I need you to explain to me why the following pull request fix the related github issue on a python repository {repo}. 
You will be given the issue description, the ground truth patch and relevant contexts.
Your job is to go across each git diff patch and explain to me the purpose of each modification and how it relates to fixing the issue.

### Issue description
{issue_description}


### Ground truth patches
{ground_truth_patch}


### Relevant contexts
{relevant_context}


Bear in mind that the ground truth patches provided passed the related unit tests and therefore its correctness is guaranteed.
Please provide your answer in the following format:

Explanation on Modification 1: [For the first modification, explain the significance of the modification location and contrast the code behavior before and after the change.]
Explanation on Modification 2: [For the second modification, explain the significance of the modification location and contrast the code behavior before and after the change.]
...
Summary: [Insert an overall summary of why the above modifications jointly fix the issue.]
"""
    return explanation_prompt


def prepare_issue_prompt(problem_stmt: str) -> str:
    """
    Given the raw problem statement, sanitize it and prepare the issue prompt.
    Args:
        problem_stmt (str): The raw problem statement.
            Assumption: the problem statement is the content of a markdown file.
    Returns:
        str: The issue prompt.
    """
    # remove markdown comments
    problem_wo_comments = re.sub(r"<!--.*?-->", "", problem_stmt, flags=re.DOTALL)
    content_lines = problem_wo_comments.split("\n")
    # remove spaces and empty lines
    content_lines = [x.strip() for x in content_lines]
    content_lines = [x for x in content_lines if x != ""]
    problem_stripped = "\n".join(content_lines)
    # add tags
    result = "<issue>" + problem_stripped + "\n</issue>"
    return result


def add_step_trigger(orig_prompt: str, is_first: bool = False) -> str:
    """
    Given the original prompt, add the trigger question for the next step.
    Args:
        orig_prompt (str): The original prompt.
        is_first (bool): Whether the trigger is for the first step.
    Returns:
        str: The prompt with trigger question.
    """
    if is_first:
        trigger = "What is the first step?"
    else:
        trigger = "What's the next step to complete the task? Be reminded that you are solving the initial issue."
    return orig_prompt + "\n" + trigger


def parse_search_actions_and_bug_locations(res_text_list):
    result_list = []
    search_pattern = r'Search Action \d+:\s*(?:`|```(?:python)?)?\s*(search_\w+\((?:[^()]*|\([^()]*\)|".*?"|\'.*?\')*?\))\s*(?:`|```)?' 
    location_pattern = re.compile(
        r"Modification Location \d+: *\n"
        r"    File Path: (?P<file_path>.*?)\n"
        r"(?:    Class Name: (?P<class_name>.*?)\n)?"
        r"(?:    Function Name: (?P<function_name>.*?)\n)?"
        r"(?:    Variable Name: (?P<variable_name>.*?)\n)?"
        r"    Reason: (?P<reason>.*?)(?:\n|$)",
        re.DOTALL
    )
    for res_text in res_text_list:
        api_json = {'API_calls':[], 'bug_locations': []}
        search_matches = re.findall(search_pattern, res_text)
        for m in search_matches:
            api_json['API_calls'].append(m)
        matches_flexible = location_pattern.finditer(res_text)
        bug_locations = [
            {
                "file_name": match.group("file_path"),
                "class_name": match.group("class_name") if match.group("class_name") else None,
                "method_name": match.group("function_name") if match.group("function_name") else None,
                "variable_name": match.group("variable_name") if match.group("variable_name") else None,
                "reasoning": match.group("reason")
            } for match in matches_flexible
        ]
        # ZZ: adhoc post process on the values, might need adjustment when new cases comes up ...
        for bug_info in bug_locations:
            for k, v in bug_info.items():
                if v: bug_info[k] = v.replace('`','')
        api_json['bug_locations'].extend(bug_locations)
        is_valid, validness_summary = is_valid_response(api_json)
        if is_valid:
            result_list.append((json.dumps(api_json),res_text)) 
        else:
            result_list.append((None,res_text)) 
    return result_list


def proxy_apis_helper(api_manager, res_text):
    selected_apis, _, proxy_threads = api_manager.proxy_apis(res_text)
    return selected_apis, res_text     


def actor_model_helper(messages):
    res_text, cost, input_tokens, output_tokens = common.ACTOR_MODEL.call(messages)
    return (res_text, cost, input_tokens, output_tokens)
    

def search_action_critic_helper(correct_call_info, critic_msg_thread):
    # ZZ: format critic prompt and send to critic models  
    critic_prompt= f"""Given the information above, can you help me to determine how relevant the following search actions is to the pull request we are dealing with?  
Specifically, you need to decide which of the following states each search action belongs to:

Relevant: The code context retrieved is DIRECTLY relevant to the pull request and having knowledge about this code context is necessary to understand why the issue happens. 
Neutral: The retrieved code context is somewhat pertaining to the pull request, however it is not a necessary information to analyse the underlying error upon.  
Unrelated: The code context retrieved is completely irrelevant and helpless to solve the pull request. It is safe to disregard this code context. 
Imprecise: The search action is not precise. This includes situations (1) Searching for code that are outside the scope of the current repoistory; (2) Searching with incorrect arguments (3) Searching with vague and imprecise arguments leading to too many results.   

### Search Actions 
{correct_call_info['collated_tool_response']}

**Please answer in the following format**:
Search Action 1: [Insert a brief analysis on the relevance of the first search action here.]
Category: [Insert only 'Relevant', 'Neutral', 'Unrelated' or 'Imprecise' here.]
Search Action 2: [Insert a brief analysis on the relevance of the second search action here.]
Category: [Insert only 'Relevant', 'Neutral', 'Unrelated' or 'Imprecise' here.]
...
"""
    relevant_thread = critic_msg_thread + [{"role": 'user', 'content':critic_prompt}]
    res_text, cost, input_tokens, output_tokens = common.CRITIC_MODEL.call(relevant_thread)
    correct_call_info["critic_prompt"] = critic_prompt
    correct_call_info["critic_response"] = res_text
    return (correct_call_info, cost, input_tokens, output_tokens)


def parse_search_action_critic_feedbacks(correct_call_info_list):
    for correct_call_info in correct_call_info_list:
        categories = re.findall(r'\*?\*?Category:\*?\*?\s(\w+)', correct_call_info["critic_response"])
        category_mapper = {
            'relevant': 1, 'neutral': 0.5, 'unrelated': 0, 'imprecise': -0.5
        }
        acc_score, success_cat_num = 0, 0 
        condition_met = False
        for cat in categories:
            if cat.lower() in category_mapper: 
                acc_score += category_mapper[cat.lower()]  
                success_cat_num += 1
            if cat.lower() == 'relevant':
                condition_met = True # ZZ: for every search action we require at least one search that is relevant 
        avg_score = acc_score/success_cat_num if success_cat_num else 0
        correct_call_info["reward"] = avg_score
        correct_call_info["selected"] = False
        correct_call_info['action_type'] = "SEARCH_CONTEXT"
        correct_call_info['condition_met'] = condition_met
    return 
        

def search_analysis_critic_helper(actor_res, collated_tool_response, critic_msg_thread, fix_patch):
    # static checking of the requires modification field will lead to many corner cases, let LLM decide instead
    # TODO add fix patch here again, let LLM decide if the search context requries modification or not . GT, Predicted, match or not  
    critic_prompt = f"""Given the information above, can you help me to determine if the following search analysis on the search results make sense or not?  
Specifically, for each search result there will be a corresponding analysis on its functionality and if it requires modification to fix the issue.
Your job is to (1) decide if the analysis on the functionality within the code execution chain is reasonable or not and (2) decide if the modification requiredness aligns with the ground truth patch or not.

### Search results
{collated_tool_response}


### Search Analysis 
{actor_res}


### Ground Truth Solution 
{fix_patch}


**Please answer in the following format**:
Search Analysis Review 1: [Insert a brief reasoning on the correctness of the first analysis on the functionality. Then analyse if ground truth solution performed any modification directly within this search results or not. Finally determine if this aligns with the prediction `Modification Required` above]
Functionality Correctness 1: [Insert only 'correct' or 'incorrect' here.]
Modification Requiredness Correctness 1: [Insert only 'correct' or 'incorrect' here.]

Search Analysis Review 2: [Insert a brief reasoning on the correctness of the second analysis on the functionality. Then analyse if ground truth solution performed any modification directly within this search results or not. Finally determine if this aligns with the prediction `Modification Required` above]
Functionality Correctness 2: [Insert only 'correct' or 'incorrect' here.]
Modification Requiredness Correctness 2: [Insert only 'correct' or 'incorrect' here.]
...
"""
    relevant_thread = critic_msg_thread + [{"role": 'user', 'content':critic_prompt}]
    res_text, cost, input_tokens, output_tokens = common.CRITIC_MODEL.call(relevant_thread)
    analysis_info = {
        "actor_response": actor_res,
        'critic_prompt': critic_prompt,
        "critic_response": res_text
    }
    return (analysis_info, cost, input_tokens, output_tokens)


def parse_search_analysis_critic_feedbacks(analysis_info_list):
    for analysis_info in analysis_info_list:
        functionality_correctness_regex = re.compile(r'\*?\*?Functionality Correctness\s*\d:+\*?\*?\s*(correct|incorrect)\s*\*?\*?', re.IGNORECASE | re.DOTALL)
        modification_requiredness_correctness_regex = re.compile(r'\*?\*?Modification Requiredness Correctness\s*\d:+\*?\*?\s*(correct|incorrect)\s*\*?\*?', re.IGNORECASE | re.DOTALL)
        func_matches = functionality_correctness_regex.findall(analysis_info["critic_response"])
        mod_matches = modification_requiredness_correctness_regex.findall(analysis_info["critic_response"])

        acc_score, action_count = 0, 0
        condition_met = True # all the analysis need to be correct  
        for functionality, bug_location in zip(func_matches, mod_matches):
            # ZZ: TODO be careful about the rewards given to functionality analysis and bug location analysis
            action_count += 1
            if functionality.strip().lower() == 'correct': 
                acc_score += 0.5
            else:
                acc_score -= 1
                condition_met = False
            if bug_location.strip().lower() == 'correct': 
                acc_score += 1
            else:
                acc_score -= 2
                condition_met = False
        reward = acc_score/action_count if action_count != 0 else 0
        analysis_info["reward"] = reward
        analysis_info["selected"] = False
        analysis_info["action_type"] = "SEARCH_ANALYSIS"
        analysis_info["condition_met"] = condition_met

    
def buggy_loc_critic_helper(buggy_loc_info, critic_msg_thread):
    # ZZ: the actor response contains location and location related reasoning, we should not need any collated search results 
    # TODO FIXME Add a logic that extracts all the modified locations and ask the critic model to check if all are found. 
    critic_prompt= f"""
Given the information above, please evaluate the following modification locations and their analyses to determine if they are correct and reasonable for generating a final patch to fix the issue.

# Modification Locations and Analyses
{buggy_loc_info['actor_response']}

Please provide your assessment in the format below:

Analysis of Modification Location 1: [Evaluate whether the first modification location identified and its associated analysis are correct. Provide a brief reasoning.]
Correctness of Modification Location 1: [Insert only 'correct' or 'incorrect' here]
Correctness of Modification Explanation 1: [Insert only 'correct' or 'incorrect' here]

Analysis of Modification Location 2: [Evaluate whether the second modification location identified and its associated analysis are correct. Provide a brief reasoning.]
Correctness of Modification Location 2: [Insert only 'correct' or 'incorrect' here]
Correctness of Modification Explanation 2: [Insert only 'correct' or 'incorrect' here]

...
"""
    relevant_thread = critic_msg_thread + [{"role": 'user', 'content':critic_prompt}]
    res_text, cost, input_tokens, output_tokens = common.CRITIC_MODEL.call(relevant_thread)
    buggy_loc_info["critic_prompt"] = critic_prompt
    buggy_loc_info["critic_response"] = res_text
    return (buggy_loc_info, cost, input_tokens, output_tokens)
    

def parse_buggy_location_critic_feedbacks(buggy_loc_info_list):
    for buggy_loc_info in buggy_loc_info_list:
        location_correctness_pattern = r"\*?\*?Correctness of Modification Location \d:+\*?\*?\s*[-\s]*(correct|incorrect)"
        explanation_correctness_pattern = r"\*?\*?Correctness of Modification Explanation \d:+\*?\*?\s*[-\s]*(correct|incorrect)"
        loc_matches = re.findall(location_correctness_pattern, buggy_loc_info["critic_response"], re.IGNORECASE)
        exp_matches = re.findall(explanation_correctness_pattern, buggy_loc_info["critic_response"], re.IGNORECASE)
        acc_score, loc_num = 0, 0 
        condition_met = True # for every location proposed, both the location and explanation need to be correct 
        for loc_mat, exp_mat in zip(loc_matches, exp_matches):
            loc_num += 1
            if loc_mat.strip().lower() == 'correct':
                if exp_mat.strip().lower() == 'correct':
                    acc_score += 1
                else:
                    acc_score += 0.5
                    condition_met = False
            else:
                acc_score -= 0.5
                condition_met = False
        avg_score = acc_score/loc_num if loc_num else 0
        buggy_loc_info["reward"] = avg_score
        buggy_loc_info["selected"] = False
        buggy_loc_info['action_type'] = "BUG_LOCATION"
        buggy_loc_info['condition_met'] = condition_met
    return 


def print_selected_search_action_results(selected_apis_json, collated_tool_response, round_no, print_callback):
    json_api_calls = selected_apis_json.get("API_calls", [])
    buggy_locations = selected_apis_json.get("bug_locations", [])
    formatted = []
    if json_api_calls:
        formatted.append("API calls:")
        for call in json_api_calls:
            formatted.extend([f"\n- `{call}`"])
    if buggy_locations:
        formatted.append("\n\nBug locations")
        for location in buggy_locations:
            s = ", ".join(f"{k}: `{v}`" for k, v in location.items())
            formatted.extend([f"\n- {s}"])
    print_acr(
        "\n".join(formatted),
        "Agent-selected API calls",
        print_callback=print_callback,
    )
    print_acr(
        collated_tool_response,
        f"context retrieval round {round_no}",
        print_callback=print_callback,
    )
    return json_api_calls, buggy_locations


def get_search_or_bug_location_prompt(round_no, repo_name):
    if round_no == 0:
        # ZZ: specify the return format in here so that we can avoid the unnecessary api calls 
        prompt = (
            "Based on the files, classes, methods, and code statements from the issue related to the bug/feature request, you can use the following search APIs to get more context of the project."
            "However, note that the search scope is limited to the issue codebase. Do not use the search tools for codebases imported or outside the issue codebase."
            f"Do not use local file_path the user described in the issue description for search, use the path that start from the issue codebase {repo_name} instead."
            "\n- search_class(class_name): Search for a class in the codebase"
            "\n- search_method_in_file(method_name, file_path): Search for a method in a given file"
            "\n- search_method_in_class(method_name, class_name): Search for a method in a given class"
            "\n- search_method(method_name): Search for a method in the entire codebase"
            "\n- search_code(code_str): Search for a code snippet in the entire codebase"
            "\n- search_code_in_file(code_str, file_path): Search for a code snippet in a given file file"
            "\n\nNote that you can use multiple search APIs in one round."
            "\nNow analyze the issue and select necessary APIs to get more context of the project. Each API call must have concrete arguments as inputs."
            "\n\nFollowing is the desired response format:"
            "\nIssue Analysis: [Provide a brief analysis on the issue, with a focus on what further contexts would we need to collect to fully understand the issue and locate the bug locations or modification locations for feature request.]"
            "\nSearch Action 1: [Insert the first search call here with concrete arguments. Do not use any path from user directory, use path starts from the issue codebase instead]"
            "\nSearch Action 2: [Insert the second search call here with concrete arguments. ]"
            "\n..."
        )
    else:
        # ZZ: TODO Do we want to make the LLM only output either the search actions or bug locations but not both ?  
        prompt = (
            "Based on your analysis, please address the following:"
            "\n1. Do we need more context? If yes, generate search API calls to gather additional project context. Leave this blank if no further context is needed."
            "\n2. Where are the locations to modify? Provide the details in the specified format below. Leave this section blank if insufficient information is available."
            "\n\nResponse Format:"
            "\nProgress Analysis: [Briefly summarize the current progress, focusing on if more context is needed to understand the cause of the issue or designs to implement the feature request. If ALL the code snippets that requires modification can be determined already, skip the search actions and proceed to the Modification Location Analysis.]"
            "\nSearch Action 1: [Include the first search call with specific arguments. Omit if sufficient context has been gathered for modifications.]"
            "\nSearch Action 2: [Include the second search call with specific arguments. Omit if sufficient context has been gathered for modifications.]"
            "\n..."
            "\nModification Location Analysis: [For the search results that requires modification, explain why they are related to the bug/feature request/issue and what modifications are needed. Leave the analysis and following location blocks **BLANK** if unknown.]"
            "\nModification Location 1: "
            "\n    File Path: [Full path to the file where the modification should take place. Required.]"
            "\n    Class Name: [Name of the class for modification. Omit if not applicable.]"
            "\n    Function Name: [Name of the function for modification. Omit if not applicable.]"
            "\n    Reason: [Explain why the code is related to the issue and what modifications are needed here.]"
            "\nModification Location 2: "
            "\n    File Path: [Full path to the file where the modification should take place. Required.]"
            "\n    Class Name: [Name of the class for modification. Omit if not applicable.]"
            "\n    Function Name: [Name of the function for modification. Omit if not applicable.]"
            "\n    Reason: [Explain why the code is related to the issue and what modifications are needed here.]"
            "\n..."
        )
    return prompt


def generate_patch_parallel(actor_msg_thread, output_dir, print_callback):
    patch_system_prompt = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
You ultimate goal is to write a patch that resolves this issue.
"""
    patch_gen_prompt = """Write a patch for the issue, based on the retrieved context.\n\nYou can import necessary libraries.\n\n
Return the patch in the format below.\n\nWithin `<file></file>`, replace `...` with actual file path.\n\nWithin `<original></original>`, replace `...` with the original code snippet from the program.\n\nWithin `<patched></patched>`, replace `...` with the fixed version of the original code. When adding orignal code and updated code, pay attention to indentation, as the code is in Python.
You can write multiple modifications if needed.

```
# modification 1
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 2
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 3
...
```
Please make sure your patch is wrapped in ``` as indicated above.
"""
    messages = deepcopy(actor_msg_thread.messages)
    patch_thread: MessageThread = MessageThread(messages=messages)
    if patch_thread.messages[0]['role'] == 'system': 
        patch_thread.messages[0]["content"] = patch_system_prompt
    patch_thread.add_user(patch_gen_prompt) 
    # We do not count the api request num for budget limit here because we enforce ground truth solution without actually running the validation test
    # therefore it is unnecessary to filter the patch generation, thus no budget control needed  
    with ThreadPoolExecutor(max_workers=globals.rejection_sampling_k) as executor:
        futures = list(executor.map(lambda x: actor_model_helper(patch_thread.to_msg()), range(globals.rejection_sampling_k)))
    res_text_list = [f[0] for f in futures]
    common.thread_cost.process_cost += sum([f[1] for f in futures])
    common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
    common.thread_cost.process_output_tokens += sum([f[3] for f in futures])
    # This step will modify the repo and have to be executed sequentially 
    applicable_patch_list, inapplicable_patch_list = [], []
    for i, patch in enumerate(res_text_list):
        raw_patch_file = pjoin(output_dir, f"agent_patch_raw_{i}")
        diff_file = pjoin(output_dir, f"extracted_patch_{i}.diff")
        with open(raw_patch_file, "w") as f:
            f.write(patch)
        extract_status, extract_msg = extract_diff_one_instance(raw_patch_file, diff_file)
        if extract_status == ExtractStatus.APPLICABLE_PATCH:
            applicable_patch_list.append(patch)
        else:
            inapplicable_patch_list.append(patch)
    print_acr(f"For {globals.rejection_sampling_k} patches generated, {len(applicable_patch_list)} are applicable patches.",
        f"solution patch generation", print_callback=print_callback)
    return applicable_patch_list, inapplicable_patch_list


def add_patch_info(actor_msg_thread, fix_patch, applicable_patch_list, inapplicable_patch_list):
    patch_info_list = []
    # ground truth patch
    patch_info_list.append({
        "actor_response": fix_patch,
        "collated_tool_response": "",
        'critic_prompt': '',
        "critic_response": '',
        "reward": 1, 
        'selected': True,
        'action_type': 'PATCH_GENERATION'
    })
    # applicable and inapplicable patches
    for patch in applicable_patch_list:
        patch_info_list.append({
            "actor_response": patch,
            "collated_tool_response": "",
            'critic_prompt': '',
            "critic_response": '',
            "reward": 0.3, 
            'selected': False,
            'action_type': 'PATCH_GENERATION'
        })
    for patch in inapplicable_patch_list:
        patch_info_list.append({
            "actor_response": patch,
            "collated_tool_response": "",
            'critic_prompt': '',
            "critic_response": '',
            "reward": -1, 
            'selected': False,
            'action_type': 'PATCH_GENERATION'
        })
    actor_msg_thread.add_rejection_sampled_messages(patch_info_list)


def is_bug_localization_finished(fix_patch, buggy_locations, critic_messages):
    formatted_bug_locations = ""
    for idx, bug_info in enumerate(buggy_locations):
        formatted_bug_locations += (
            f'Nominated Modification Location {idx+1}: \n'
            f"file_name: {bug_info['file_name']}\n"
            f"class_name: {bug_info['class_name']}\n" if 'class_name' in bug_info else ''
            f"method_name: {bug_info['method_name']}\n" if 'method_name' in bug_info else ''
            f"variable_name: {bug_info['variable_name']}\n" if 'variable_name' in bug_info else ''
            f"analysis: {bug_info['reasoning']}\n"
        )
    # TODO Patch includes modification on existing code and addition of non-existing code
    # existing code modification can be nominated, but helper functions/new vars/new classes are hard to nominate exact locations
    # Design the prompt such that it can handle cases mentioned above  
    critic_prompt = f"""Given the analysis above, please compare the nominated modification locations and the actual modifications in the ground truth fix patch.
Your job is to check if for every modification locations from the ground truth, there exists a corresponding location in the nominated list or not. 
Note for the ground truth fix patch, it might contain some addition of helper functions, new classes or new variables that can be added in a number of different candidate places.
For these additions, check if these changes are subsidiary changes of some nominated modification locations.   

### Ground Truth Fix Patch
{fix_patch}


### Nominated Modification Locations   
{formatted_bug_locations}


Below is the desired format for response:
Modification Location 1: [Insert the first ground truth modification locations here]
Modification Location 1 Analysis: [Analyse if the first ground truth modification is nominated directly in the locations above or can be contained within a proposed modification.]
Matched Nominated Location 1 : [Insert the matched nominated location or N/A here if no matched location found]

Modification Location 2: [Insert the second ground truth modification locations here]
Modification Location 2 Analysis: [Analyse if the second ground truth modification is nominated directly in the locations above or can be contained within a proposed modification.]
Matched Nominated Location 2 : [Insert the matched nominated location or N/A here if no matched location found] 

...
"""
    relevant_thread = critic_messages + [{"role": 'user', 'content':critic_prompt}]
    res_text, cost, input_tokens, output_tokens = common.CRITIC_MODEL.call(relevant_thread)
    # parse the critic results and check if any N/A exists 
    # TODO FIXME check if the prompts needs improvement
    all_bug_found = not ('N/A' in res_text or 'n/a' in res_text)
    return all_bug_found

def start_conversation_round_stratified(
    output_dir: str,
    actor_msg_thread: MessageThread,
    critic_msg_thread: MessageThread,
    api_manager: ProjectApiManager,
    repo_name: str='',
    print_callback: Callable[[dict], None] | None = None,
    fix_patch: str=''
) -> bool:
    """
    This version uses formatted response instead of using the OpenAI function calling.
    Advantage is that multiple API calls can be made in a single round.
    """
    api_request_count = 0
    # TODO: FIXME setup budget control here. once met break anyway
    for round_no in range(globals.conv_round_limit):
        api_manager.start_new_tool_call_layer()

        conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
        # save current state before starting a new round
        actor_msg_thread.save_to_file(conversation_file)

        print_banner(f"CONTEXT RETRIEVAL ROUND {round_no}")
        prompt = get_search_or_bug_location_prompt(round_no, repo_name)
        actor_msg_thread.add_user(prompt)
        print_acr(
            prompt,
            f"context retrieval round {round_no}",
            print_callback=print_callback,
        )
        search_action_condition_met = False
        combined_info_list = [] # list for the repeated sampling 
        while not search_action_condition_met:
            # ZZ: perform batch inference here so that we can apply rejection sampling, note res_text is a list and need special handling
            with ThreadPoolExecutor(max_workers=globals.rejection_sampling_k) as executor:
                futures = list(executor.map(lambda x: actor_model_helper(actor_msg_thread.to_msg()), range(globals.rejection_sampling_k)))
            res_text_list = [f[0] for f in futures]
            common.thread_cost.process_cost += sum([f[1] for f in futures])
            common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
            common.thread_cost.process_output_tokens += sum([f[3] for f in futures])
            api_request_count += globals.rejection_sampling_k
            if api_request_count >= globals.total_request_num: break
            
            # ZZ: replace the proxy api call with simple parsing to extract search calls and potential bug locations  
            parsed_results = parse_search_actions_and_bug_locations(res_text_list)
            # ZZ: Categorize each search attempt into correct and incorrect calls 
            correct_tool_calls, incorrect_tool_calls, = [], []
            correct_buggy_locations, incorrect_buggy_locations = [], []   
            for parsed_apis, res_text in parsed_results:
                apis_json = json.loads(parsed_apis) if parsed_apis is not None else {}
                curr_json_api_calls = apis_json.get("API_calls", [])
                curr_buggy_locations = apis_json.get("bug_locations", [])
                if parsed_apis is None:
                    incorrect_tool_calls.append({
                        "actor_response": res_text,
                        'parsed_apis': parsed_apis,
                        "collated_tool_response": "",
                        'critic_prompt': '',
                        "critic_response": DEFAULT_CRITIC_RESPONSE, # ZZ: provide a default critic feedback to failed actions 
                        "reward": -1,  # ZZ: by default we give -1 to actions not following the desired format ?
                        'selected': False,
                        'action_type': 'FAILED'
                    })
                elif curr_json_api_calls:
                    # ZZ: for search actions successfully parsed, we should collect related contexts, send to critic models to get feedbacks  
                    collated_tool_response = ""
                    for api_call in curr_json_api_calls:
                        func_name, func_args = parse_function_invocation(api_call)

                        arg_spec = inspect.getfullargspec(getattr(SearchManager, func_name))
                        arg_names = arg_spec.args[1:]  # first parameter is self

                        assert len(func_args) == len(
                            arg_names
                        ), f"Number of argument is wrong in API call: {api_call}"

                        kwargs = dict(zip(arg_names, func_args))
                        intent = FunctionCallIntent(func_name, kwargs, None)
                        tool_output, _, _ = api_manager.dispatch_intent(intent, actor_msg_thread)

                        collated_tool_response += f"Result of {api_call}:\n\n"
                        collated_tool_response += tool_output + "\n\n"
                    # fill in the critic response and reward later
                    correct_tool_calls.append({
                        "actor_response": res_text,
                        "parsed_apis": parsed_apis,
                        "collated_tool_response": collated_tool_response,
                    })
                else:
                    # ZZ: We only perform bug location examanations when no search actions are needed!
                    # if no search action needed, we should examine the buggy locations and provide the buggy locations to the model
                    for bug_location in curr_buggy_locations:
                        tool_output, *_ = search_for_bug_location(
                            api_manager, None, bug_location
                        )
                        collated_tool_response += f"\n\n{tool_output}\n"
                    if (
                        "Unknown function" not in collated_tool_response
                        and "Could not" not in collated_tool_response
                    ):
                        correct_buggy_locations.append({
                            "actor_response": res_text,
                            "parsed_apis": parsed_apis,
                            "collated_tool_response": collated_tool_response
                        })
                    else:
                        # TODO FIXME should we worry about the default feedback and reward here ?
                        incorrect_buggy_locations.append({
                            "actor_response": res_text,
                            'parsed_apis': parsed_apis,
                            "collated_tool_response": collated_tool_response,
                            'critic_prompt': '',
                            "critic_response": DEFAULT_CRITIC_RESPONSE, # ZZ: provide a default critic feedback to failed actions 
                            "reward": -1,  # ZZ: by default we give -1 to actions not following the desired format ?
                            'selected': False,
                            'action_type': 'FAILED'
                        })
            # ZZ: TODO FIXME What if some do search and some propose bug locations only ? or some search + correct bug locations, How to compare reward values ?
            if correct_tool_calls:
                with ThreadPoolExecutor(max_workers=len(correct_tool_calls)) as executor:
                    futures = list(executor.map(lambda correct_call_info: search_action_critic_helper(correct_call_info, critic_msg_thread.to_msg()), correct_tool_calls))
                common.thread_cost.process_cost += sum([f[1] for f in futures])
                common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
                common.thread_cost.process_output_tokens += sum([f[3] for f in futures])
                parse_search_action_critic_feedbacks(correct_tool_calls)
                api_request_count += len(correct_tool_calls)
                
            if correct_buggy_locations:
                with ThreadPoolExecutor(max_workers=len(correct_buggy_locations)) as executor:
                    futures = list(executor.map(lambda buggy_loc_info: buggy_loc_critic_helper(buggy_loc_info, critic_msg_thread.to_msg()), correct_buggy_locations))
                common.thread_cost.process_cost += sum([f[1] for f in futures])
                common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
                common.thread_cost.process_output_tokens += sum([f[3] for f in futures])
                parse_buggy_location_critic_feedbacks(correct_buggy_locations)
                api_request_count += len(correct_buggy_locations)
                # TODO FIXME Save the buggy locations and at some point judge if all the locations that requires modification are collected ! 

            sorted_correct_tool_calls = sorted(correct_tool_calls+correct_buggy_locations, key=lambda x: x['reward'], reverse=True) 
            # ZZ: save the rejection sampling actor results and critic feedbacks
            combined_info_list.extend(correct_tool_calls + incorrect_tool_calls + correct_buggy_locations + incorrect_buggy_locations)
            # make sure 1. the search actions have at least one relevant 2. the bug location and reasoning are correct
            if sorted_correct_tool_calls[0]['condition_met']:
                if sorted_correct_tool_calls[0]['action_type'] == "BUG_LOCATION":
                    # For responses that only nominates bug locations, we require it to nominate every bug locations. Otherwise keeps sampling until the model wants to explore some other RELEVANT contexts with search actions
                    # TODO: This might be a very strict condition, be very careful on the prompts and reasoning behind
                    json.loads(sorted_correct_tool_calls[0]['parsed_apis'])
                    buggy_locations = selected_apis_json.get("bug_locations", [])
                    search_action_condition_met = is_bug_localization_finished(fix_patch, buggy_locations, critic_msg_thread.to_msg())
                else:
                    # For search actions, as long as the search contexts are relevant, we can proceed to search results analysis
                    search_action_condition_met = True
            if api_request_count >= globals.total_request_num: break
            
        sorted_correct_tool_calls[0]['selected'] = True
        actor_msg_thread.add_model(sorted_correct_tool_calls[0]['actor_response'], tools=[])
        actor_msg_thread.add_user(sorted_correct_tool_calls[0]['collated_tool_response'])
        print_retrieval(sorted_correct_tool_calls[0]['actor_response'], f"round {round_no}", print_callback=print_callback)
        
        actor_msg_thread.add_rejection_sampled_messages(combined_info_list)
        selected_apis_json = json.loads(sorted_correct_tool_calls[0]['parsed_apis'])
        json_api_calls, buggy_locations = print_selected_search_action_results(selected_apis_json, 
            sorted_correct_tool_calls[0]['collated_tool_response'], round_no, print_callback)
        
        if api_request_count >= globals.total_request_num:
            print_acr(f"Finish the trajectory collection due to api request number exceeds the budget {globals.total_request_num} ...", "Trajectory Collection Finished", print_callback=print_callback) 
            break
        
        # TODO FIXME add the check that if all the contexts required are 
        # run parallel generation here and then sequentially apply the changes, then save the whole trajectory and finish the whole pipeline here .
        if sorted_correct_tool_calls[0]['action_type'] == "BUG_LOCATION":
            # If the selected action is bug locations, it should 
            print_banner("PATCH GENERATION")
            logger.debug("Gathered enough information. Invoking write_patch.")
            
            applicable_patch_list, inapplicable_patch_list = generate_patch_parallel(actor_msg_thread, output_dir, print_callback)
            logger.info("Ending workflow.")
            # save all the patches along with the ground truth into the actor thread 
            add_patch_info(actor_msg_thread, fix_patch, applicable_patch_list, inapplicable_patch_list)
            conversation_file = pjoin(output_dir, f"final_trajectory.json")
            actor_msg_thread.save_to_file(conversation_file)
            break 
        # summarize the contexts collected, update the understanding of the issues, then proceed to each context. Replace bug location with requires modification    
        msg = """Let's analyze collected context in the following desired format:
Contexts Summary: [Summarize the contexts collected on what they do and how they contribute to the execution flow. Then reflect on how they update your knowledge on the root cause of the issue or design for feature request.]

Code Explanation 1: [For the first search result, briefly explain the functionalities/logic of the code snippet returned.]
Code Relevance 1: [For the first search result, analyze if we need any modification here to fix the issue or implement the feature request. If so, what is it?]   
Modification Required 1: [Insert `true` or `false` here ONLY] 

Code Explanation 2: [For the second search result, briefly explain the functionalities/logic of the code snippet returned.]
Code Relevance 2: [For the second search result, analyze if we need any modification here to fix the issue or implement the feature request. If so, what is it?]   
Modification Required 2: [Insert `true` or `false` here ONLY] 

...
"""
        actor_msg_thread.add_user(msg)
        print_acr(
            msg, f"context retrieval round {round_no}", print_callback=print_callback
        )
        search_analysis_condition_met = False
        search_analysis_info_list = []
        while not search_analysis_condition_met:
            with ThreadPoolExecutor(max_workers=globals.rejection_sampling_k) as executor:
                futures = list(executor.map(lambda x: actor_model_helper(actor_msg_thread.to_msg()), range(globals.rejection_sampling_k)))
            res_text_list = [f[0] for f in futures]
            common.thread_cost.process_cost += sum([f[1] for f in futures])
            common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
            common.thread_cost.process_output_tokens += sum([f[3] for f in futures])
            api_request_count += globals.rejection_sampling_k
            if api_request_count >= globals.total_request_num: break

            with ThreadPoolExecutor(max_workers=globals.rejection_sampling_k) as executor:
                futures = list(executor.map(lambda res_text: search_analysis_critic_helper(res_text, 
                    sorted_correct_tool_calls[0]["collated_tool_response"], critic_msg_thread.to_msg(), fix_patch), res_text_list))
            common.thread_cost.process_cost += sum([f[1] for f in futures])
            common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
            common.thread_cost.process_output_tokens += sum([f[3] for f in futures])
            api_request_count += globals.rejection_sampling_k

            search_analysis_list = [f[0] for f in futures]
            parse_search_analysis_critic_feedbacks(search_analysis_list)
            search_analysis_info_list.extend(search_analysis_list)
            sorted_search_analysis_list = sorted(search_analysis_info_list, key=lambda x: x['reward'], reverse=True) 
            search_analysis_condition_met = sorted_search_analysis_list[0]['condition_met']
            if api_request_count >= globals.total_request_num: break
            
        # ZZ: select the best search analysis 
        sorted_search_analysis_list[0]['selected'] = True
        actor_msg_thread.add_model(sorted_search_analysis_list[0]['actor_response'], tools=[])        
        print_retrieval(sorted_search_analysis_list[0]['actor_response'], f"round {round_no}", print_callback=print_callback)
        # ZZ: save the rejection sampling actor results and critic feedbacks
        actor_msg_thread.add_rejection_sampled_messages(search_analysis_info_list)
        if api_request_count >= globals.total_request_num: 
            print_acr(f"Finish the trajectory collection due to api request number exceeds the budget {globals.total_request_num} ...", "Trajectory Collection Finished", print_callback=print_callback) 
            break

    return True


def search_for_bug_location(
    api_manager: ProjectApiManager,
    msg_thread: MessageThread,
    bug_location: dict[str, str],
) -> tuple[str, str, bool]:
    found = False
    # ZZ: TODO FIXME add the search for variable names here ...
    file_name = bug_location.get("file_name")
    method_name = bug_location.get("method_name")
    class_name = bug_location.get("class_name")

    assert method_name or class_name, f"Invalid bug location: {bug_location}"

    call_result = None

    def call_function(func_name: str, kwargs: dict[str, str]) -> None:
        nonlocal found, call_result

        intent = FunctionCallIntent(func_name, kwargs, None)
        call_result = api_manager.dispatch_intent(intent, msg_thread)
        _, _, call_is_ok = call_result
        found |= call_is_ok

    if (not found) and method_name and class_name:
        kwargs = {"method_name": method_name, "class_name": class_name}
        call_function("search_method_in_class", kwargs)

    if (not found) and method_name and file_name:
        kwargs = {"method_name": method_name, "file_name": file_name}
        call_function("search_method_in_file", kwargs)

    if (not found) and class_name and file_name:
        kwargs = {"class_name": class_name, "file_name": file_name}
        call_function("search_class_in_file", kwargs)

    if (not found) and class_name:
        kwargs = {"class_name": class_name}
        call_function("get_class_full_snippet", kwargs)

    if (not found) and method_name:
        kwargs = {"method_name": method_name}
        call_function("search_method", kwargs)

    assert call_result

    return call_result


def dump_tool_call_layers_to_file(
    tool_call_layers: list[dict], output_dir: str
) -> None:
    """Dump the layers of tool calls to a file."""
    tool_call_file = pjoin(output_dir, "tool_call_layers.json")
    with open(tool_call_file, "w") as f:
        json.dump(tool_call_layers, f, indent=4)


def start_conversation_round_state_machine(
    output_dir: str,
    msg_thread: MessageThread,
    api_manager: ProjectApiManager,
    start_round_no: int = 0,
) -> bool:
    """
    Start the actual rounds of conversations with model.

    Args:
        output_dir (str): Path to the output directory.
        msg_thread (MessageThread): The message thread to be used.
        api_manager (ProjectApiManager): The API manager to be used.
        start_round_no (int): The round number to start with.
    """
    round_no = start_round_no
    for round_no in range(start_round_no, globals.conv_round_limit + 1):
        conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
        # save current state before starting a new round
        msg_thread.save_to_file(conversation_file)
        log_and_cprint(
            f"\n========== Conversation Round {round_no} ==========", style="red bold"
        )
        log_and_print(f"{colored('Current message thread:', 'green')}\n{msg_thread}")

        allowed_tools = api_manager.next_tools()
        # TODO: configure the list of tools based on state machine
        tools = ProjectApiManager.get_full_funcs_for_openai(allowed_tools)

        log_and_cprint(f"Current tool state: {api_manager.curr_tool}", style="yellow")
        log_and_cprint(f"Allowed next tool states: {allowed_tools}", style="yellow")

        # create a new iteration of conversation
        res_text, raw_tool_calls, func_call_intents, *_ = common.ACTOR_MODEL.call(
            msg_thread.to_msg(), tools=tools
        )
        log_and_print(
            f"{colored('This roud model response (text):', 'blue')} {res_text}"
        )
        # model can decide whether to create a function call
        if len(func_call_intents) == 1:
            # good case in which we can check function call
            func_call_intent: FunctionCallIntent = func_call_intents[0]
            log_and_print(
                f"{colored('This round model response (function call):', 'blue')} {func_call_intent}"
            )
            # dispatch this function call
            this_model_response = res_text
            this_model_tools = raw_tool_calls
            # add previous call information to user message
            tool_output, summary, _ = api_manager.dispatch_intent(
                func_call_intent, msg_thread
            )
        else:
            # no function call, let's force the model to make one
            this_model_tools = []
            this_model_response = res_text
            tool_output = ""
            summary = "There is no function call in your previous response. Make sure you include one function call. "

        next_user_message = add_step_trigger(summary)

        # form message thread for next round. should include what the model said as well
        msg_thread.add_model(this_model_response, this_model_tools)
        if this_model_tools:
            tool_call_id = this_model_tools[0].id
            msg_thread.add_tool(tool_output, tool_call_id)
            msg_thread.add_user(next_user_message)
        else:
            msg_thread.add_user(next_user_message)

        if len(func_call_intents) == 1:
            func_call_name = func_call_intents[0].func_name
            if func_call_name == "write_patch":
                log_and_print("Ending workflow. write_patch has been invoked.")
                break

        log_and_print("Going to next round ..........")
    else:
        log_and_print("Too many rounds. Try writing patch anyway.")
        write_patch_intent = FunctionCallIntent("write_patch", {}, None)
        api_manager.dispatch_intent(write_patch_intent, msg_thread)

    round_no += 1

    # if we end the workflow normally, there is one more round of conversation to store
    conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
    msg_thread.save_to_file(conversation_file)
    return True


def get_patch_explanation(repo_name:str, problem_statement:str, fix_patch:str, api_manager: ProjectApiManager) -> tuple[str, str]:    
    patch_info_list = parse_git_patch(fix_patch)
    relevant_context = ""
    for patch_info in patch_info_list:
        file_name = patch_info["original_file_path"]
        hunk_context = patch_info["hunk_context"]
        # ZZ: TODO expand this to more cases ....
        if hunk_context.startswith("def "):
            method_name = hunk_context.split("def ")[1].split("(")[0]
            intent = FunctionCallIntent("search_method_in_file", {
                "method_name": method_name, "file_name": file_name}, None)
            detailed_result, succint_result, found = api_manager.dispatch_intent(intent, None)
        elif hunk_context.startswith("class "):
            class_name = hunk_context.split("class ")[1].split("(")[0]
            intent = FunctionCallIntent("search_class_in_file", {
                "class_name": class_name, "file_name": file_name}, None)
            detailed_result, succint_result, found = api_manager.dispatch_intent(intent, None)
        elif not hunk_context:
            # TODO can we support search by chunk here ? 
            intent = FunctionCallIntent("search_code_in_file", {
                "code_str": patch_info['original_hunk_lines'][0], "file_name": file_name}, None)
            detailed_result, succint_result, found = api_manager.dispatch_intent(intent, None)
        else:
            # TODO: simply ignore this ?
            continue
        relevant_context += detailed_result
    exp_prompt = prepare_explanation_prompt(repo_name, problem_statement, fix_patch, relevant_context)
    explanation_result, *_ = common.CRITIC_MODEL.call([{"role": "user", "content": exp_prompt}])
    return exp_prompt, explanation_result


def run_one_task(
    output_dir: str,
    api_manager: ProjectApiManager,
    problem_stmt: str,
    fix_patch: str,
    repo_name: str,
    print_callback: Callable[[dict], None] | None = None,
) -> bool:
    """
    Main entry point to run inference on one task with rejection sampling to collect success trajectories.
    Args:
        output_dir (str): Path to the output directory.
        api_manager (ProjectApiManager): The already-initialized API manager.
        problem_stmt (str): The original problem statement submitted to the task issue.
    """
    # ZZ: Get explanation for critic model first.
    print_banner("Collecting explanations for the issue")
    exp_prompt, explanation_result = get_patch_explanation(repo_name, problem_stmt, fix_patch, api_manager)
    critic_msg_thread = MessageThread()
    critic_msg_thread.add_user(exp_prompt)
    critic_msg_thread.add_model(explanation_result, tools=[])
    print_acr(exp_prompt, f"Gournd Truth Explanation Prompt", print_callback=print_callback,)
    print_acr(explanation_result, f"Gournd Truth Explanation Result", print_callback=print_callback,)
    
    print_banner("Starting AutoCodeRover on the following issue")
    print_issue(problem_stmt)
    actor_msg_thread = MessageThread()   

    system_prompt = SYSTEM_PROMPT
    if (not globals.enable_layered) and common.ACTOR_MODEL.parallel_tool_call:
        # these models support parallel tool calls, let's try to make them not do it
        system_prompt += " In your response, DO NOT make more than one tool call."

    actor_msg_thread.add_system(system_prompt)
    original_prompt = prepare_issue_prompt(problem_stmt)
    actor_msg_thread.add_user(original_prompt)

    # Add another user message about fault localization
    if globals.enable_sbfl:
        localization_result, _, _ = api_manager.fault_localization()
        localization_prompt = "An external analysis tool has been deployed to identify the suspicious code to be fixed. You can choose to use the results from this tool, if you think they are useful."
        localization_prompt += "The tool output is as follows:\n"
        localization_prompt += localization_result
        actor_msg_thread.add_user(localization_prompt)

    if globals.enable_layered:
        return start_conversation_round_stratified(
            output_dir, actor_msg_thread, critic_msg_thread, api_manager, repo_name=repo_name, print_callback=print_callback, fix_patch=fix_patch
        )
    else:
        # ZZ: let's ignore this branch since it necessitate more rounds of interactions and more tokens ... 
        return start_conversation_round_state_machine(
            output_dir, actor_msg_thread, api_manager
        )


# NOTE: deprecated
def continue_task_from_cache(
    cache_path: str, output_dir: str, api_manager: ProjectApiManager
) -> bool:
    """
    Run inference on one task, but load conversation history from cache.
    Args:
        cache_path (str): Path to the old conversation history file.
        output_dir (str): Path to the output directory.
        api_manager (ProjectApiManager): The already-initialized API manager.
    """
    # (1) load the existing message thread
    msg_thread = MessageThread.load_from_file(cache_path)
    completed_round_no = msg_thread.get_round_number()

    # (2) start the actual workflow
    return start_conversation_round_state_machine(
        output_dir, msg_thread, api_manager, start_round_no=completed_round_no
    )
