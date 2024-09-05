import inspect
import json
import re
from collections.abc import Callable
from os.path import join as pjoin
from pathlib import Path

from loguru import logger
from termcolor import colored
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

# FIXME: the system prompt should be different for stratified/state machine.
SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Your task is to invoke a few search API calls to gather buggy information, then write patches to solve the issues.
"""

# FIXME: Adjust the response according to the parsing rule of search actions 
DEFAULT_CRITIC_RESPONSE = """Failed to parse the search actions out of the response. The search action generated might not be following the instructions given. 
Please check your response and follow the **exact** format desired.    
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
    categories = re.findall(r'Category: (\w+)', res_text)
    # ZZ: TODO How do we assign scores to the actions. Average or Best? What is the score for each category?
    category_mapper = {
        'relevant': 1, 'neutral': 0.5, 'unrelated': 0, 'imprecise': -0.2
    }
    acc_score, success_cat_num = 0, 0 
    for cat in categories:
        if cat.lower() in category_mapper: 
            acc_score += category_mapper[cat.lower()]  
            success_cat_num += 1
    avg_score = acc_score/success_cat_num if success_cat_num else 0
    correct_call_info["critic_prompt"] = critic_prompt
    correct_call_info["critic_response"] = res_text
    correct_call_info["reward"] = avg_score
    correct_call_info["selected"] = False
    return (correct_call_info, cost, input_tokens, output_tokens)
    
    
def search_analysis_critic_helper(actor_res, selected_critic_info, critic_msg_thread):
    critic_prompt = f"""Given the information above, can you help me to determine if the following search analysis on the search results make sense or not?  
Specifically, for each search result there will be a corresponding analysis on its functionality and if it is the bug location.
Your job is to (1) decide if the analysis on the functionality within the code execution chain is reasonable or not and (2) decide if the determination of bug location is correct or not.

### Search Analysis 
{actor_res}

**Please answer in the following format**:
Search Analysis Review 1: [Insert a brief reasoning on the correctness of the first analysis on the functionality and bug location here.]
Functionality Correctness 1: [Insert only 'correct' or 'incorrect' here.]
Bug Location Correctness 1: [Insert only 'correct' or 'incorrect' here.]

Search Analysis Review 2: [Insert a brief reasoning on the correctness of the second analysis on the functionality and bug location here.]
Functionality Correctness 2: [Insert only 'correct' or 'incorrect' here.]
Bug Location Correctness 2: [Insert only 'correct' or 'incorrect' here.]
...
"""
    relevant_thread = critic_msg_thread + [
        {"role": 'user', 'content':selected_critic_info['critic_prompt']},
        {"role": 'assistant', 'content':selected_critic_info['critic_response']}
        {"role": 'user', 'content':critic_prompt}
    ]
    res_text, cost, input_tokens, output_tokens = common.CRITIC_MODEL.call(relevant_thread)
    pattern = r'Functionality Correctness \d+: (\w+)\nBug Location Correctness \d+: (\w+)'
    matches = re.findall(pattern, res_text)
    acc_score, action_count = 0
    for i, (functionality, bug_location) in enumerate(matches, start=1):
        # ZZ: TODO be careful about the rewards given to functionality analysis and bug location analysis 
        # ZZ: if any of the bug location is incorrect ... should we simply disregard this analysis ???
        action_count += 1
        if functionality.strip().lower() == 'correct': 
            acc_score += 0.5
        else:
            acc_score -= 1
        if bug_location.strip().lower() == 'correct': 
            acc_score += 1
        else:
            acc_score -= 2 
    reward = acc_score/action_count if action_count != 0 else 0
    analysis_info = {
        "actor_response": actor_res,
        'critic_prompt': critic_prompt,
        "critic_response": res_text, 
        "reward": reward,  
        'selected': False
    }
    return (analysis_info, cost, input_tokens, output_tokens)
    

    
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
    

def start_conversation_round_stratified(
    output_dir: str,
    actor_msg_thread: MessageThread,
    critic_msg_thread: MessageThread,
    api_manager: ProjectApiManager,
    start_round_no: int = 0,
    print_callback: Callable[[dict], None] | None = None,
) -> bool:
    """
    This version uses json data to process API calls, instead of using the OpenAI function calling.
    Advantage is that multiple API calls can be made in a single round.
    """
    # ZZ: TODO specify the return format in here so that we can avoid the unnecessary api calls 
    prompt = (
        "Based on the files, classes, methods, and code statements from the issue related to the bug, you can use the following search APIs to get more context of the project."
        "\n- search_class(class_name: str): Search for a class in the codebase"
        "\n- search_method_in_file(method_name: str, file_path: str): Search for a method in a given file"
        "\n- search_method_in_class(method_name: str, class_name: str): Search for a method in a given class"
        "\n- search_method(method_name: str): Search for a method in the entire codebase"
        "\n- search_code(code_str: str): Search for a code snippet in the entire codebase"
        "\n- search_code_in_file(code_str: str, file_path: str): Search for a code snippet in a given file file"
        "\n\nNote that you can use multiple search APIs in one round."
        "\n\nNow analyze the issue and select necessary APIs to get more context of the project. Each API call must have concrete arguments as inputs."
    )
    actor_msg_thread.add_user(prompt)

    round_no = start_round_no

    round_count = range(start_round_no, globals.conv_round_limit + 1)

    for round_no in round_count:
        api_manager.start_new_tool_call_layer()

        conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
        # save current state before starting a new round
        actor_msg_thread.save_to_file(conversation_file)

        print_banner(f"CONTEXT RETRIEVAL ROUND {round_no}")

        print_acr(
            prompt,
            f"context retrieval round {start_round_no}",
            print_callback=print_callback,
        )
        # ZZ: perform batch inference here so that we can apply rejection sampling, note res_text is a list and need special handling
        with ThreadPoolExecutor(max_workers=globals.rejection_sampling_k) as executor:
            futures = list(executor.map(lambda x: actor_model_helper(actor_msg_thread.to_msg()), range(globals.rejection_sampling_k)))
        res_text_list = [f[0] for f in futures]
        common.thread_cost.process_cost += sum([f[1] for f in futures])
        common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
        common.thread_cost.process_output_tokens += sum([f[3] for f in futures])
        
        # ZZ: proxy api extract the search action in json format. We make a parallel call here for all the different res_texts generated 
        with ThreadPoolExecutor(max_workers=globals.rejection_sampling_k) as executor:
            futures = list(executor.map(lambda res_text: proxy_apis_helper(api_manager, res_text), res_text_list))
            # ZZ: TODO Modify the search action call and therefore get rid of this proxy apis call. This is indirect and unnecessary !
        
        # ZZ: Categorize each search attempt into correct and incorrect calls 
        correct_tool_calls, incorrect_tool_calls = [], [] 
        for parsed_apis, res_text in futures:
            if parsed_apis is None:
                incorrect_tool_calls.append({
                    "actor_response": res_text,
                    'parsed_apis': parsed_apis,
                    "collated_tool_response": "",
                    'critic_prompt': '',
                    "critic_response": DEFAULT_CRITIC_RESPONSE, # ZZ: provide a default critic feedback to failed actions 
                    "reward": -1,  # ZZ: by default we give -1 to actions not following the desired format ?
                    'selected': False
                })
            else:
                # ZZ: for search actions successfully parsed, we should collect related contexts, send to critic models to get feedbacks  
                apis_json = json.loads(parsed_apis)
                curr_json_api_calls = apis_json.get("API_calls", [])
                # prepare response from tools
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
        # ZZ: make parallel calls to critic models 
        with ThreadPoolExecutor(max_workers=len(correct_tool_calls)) as executor:
            futures = list(executor.map(lambda correct_call_info: search_action_critic_helper(correct_call_info, critic_msg_thread.to_msg()), correct_tool_calls))
        common.thread_cost.process_cost += sum([f[1] for f in futures])
        common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
        common.thread_cost.process_output_tokens += sum([f[3] for f in futures])
        
        # ZZ: select the best search action to proceed
        sorted_correct_tool_calls = sorted(correct_tool_calls, key=lambda x: x['reward'], reverse=True) 
        sorted_correct_tool_calls[0]['selected'] = True
        actor_msg_thread.add_model(sorted_correct_tool_calls[0]['actor_response'], tools=[])
        actor_msg_thread.add_user(sorted_correct_tool_calls[0]['collated_tool_response'])
        print_retrieval(sorted_correct_tool_calls[0]['actor_response'], f"round {round_no}", print_callback=print_callback)
        
        # ZZ: save the rejection sampling actor results and critic feedbacks
        actor_msg_thread.add_rejection_sampled_messages(correct_tool_calls+incorrect_tool_calls)

        # ZZ: TODO Do we need to print this ? Or delete this logic ...
        selected_apis_json = json.loads(sorted_correct_tool_calls[0]['parsed_apis'])
        json_api_calls, buggy_locations = print_selected_search_action_results(selected_apis_json, 
            sorted_correct_tool_calls[0]['collated_tool_response'], round_no, print_callback)
        
        # collected enough information to write patch
        if buggy_locations and (not json_api_calls):
            collated_tool_response = "Here is the code in buggy locations:\n\n"
            # provide the buggy locations to the model
            for bug_location in buggy_locations:
                tool_output, *_ = search_for_bug_location(
                    api_manager, actor_msg_thread, bug_location
                )
                collated_tool_response += f"\n\n{tool_output}\n"

            if (
                "Unknown function" not in collated_tool_response
                and "Could not" not in collated_tool_response
            ):
                actor_msg_thread.add_user(collated_tool_response)
                print_banner("PATCH GENERATION")
                logger.debug("Gathered enough information. Invoking write_patch.")
                print_acr(
                    collated_tool_response,
                    "patch generation round 1",
                    print_callback=print_callback,
                )
                break

            msg = "The buggy locations is not precise. You may need to check whether the arguments are correct and search more information."
            actor_msg_thread.add_user(msg)
            print_acr(
                msg,
                f"context retrieval round {round_no}",
                print_callback=print_callback,
            )
            continue

        
        # ZZ: TODO specify the analysis directions and critic scoring fields 
        msg = """Let's analyze collected context in the following desired format:
Code Explanation 1: [For the first search result, briefly explain its functionalities.]
Code Relevance 1: [For the first search result, what role does it play in the execution chain of the issue described? Is the bug directly located within this search result rather than nested in one of its related function calls?]   
Is Bug Location 1: [Insert `true` or `false` here ONLY] 

Code Explanation 2: [For the second search result, briefly explain its functionalities.]
Code Relevance 2: [For the second search result, what role does it play in the execution chain of the issue described? Is the bug directly located within this search result rather than nested in one of its related function calls?]   
Is Bug Location 2: [Insert `true` or `false` here ONLY] 

...
"""
        actor_msg_thread.add_user(msg)
        print_acr(
            msg, f"context retrieval round {round_no}", print_callback=print_callback
        )
        with ThreadPoolExecutor(max_workers=globals.rejection_sampling_k) as executor:
            futures = list(executor.map(lambda x: actor_model_helper(actor_msg_thread.to_msg()), range(globals.rejection_sampling_k)))
        res_text_list = [f[0] for f in futures]
        common.thread_cost.process_cost += sum([f[1] for f in futures])
        common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
        common.thread_cost.process_output_tokens += sum([f[3] for f in futures])

        # TODO: Do we really need to analyse the bug locations ? Or do we need k analysis ? 
        with ThreadPoolExecutor(max_workers=globals.rejection_sampling_k) as executor:
            futures = list(executor.map(lambda res_text: search_analysis_critic_helper(res_text, 
                sorted_correct_tool_calls[0], critic_msg_thread.to_msg()), res_text_list))
        common.thread_cost.process_cost += sum([f[1] for f in futures])
        common.thread_cost.process_input_tokens += sum([f[2] for f in futures])
        common.thread_cost.process_output_tokens += sum([f[3] for f in futures])
        
        # ZZ: select the best search analysis 
        search_analysis_info_list = [f[0] for f in futures]
        sorted_search_analysis_list = sorted(search_analysis_info_list, key=lambda x: x['reward'], reverse=True) 
        sorted_search_analysis_list[0]['selected'] = True
        actor_msg_thread.add_model(sorted_search_analysis_list[0]['actor_response'], tools=[])        
        print_retrieval(res_text, f"round {round_no}", print_callback=print_callback)

        # ZZ: save the rejection sampling actor results and critic feedbacks
        actor_msg_thread.add_rejection_sampled_messages(search_analysis_info_list)

        if round_no < globals.conv_round_limit:
            msg = (
                "Based on your analysis, answer below questions:"
                "\n- do we need more context: construct search API calls to get more context of the project. (leave it empty if you don't need more context)"
                "\n- where are bug locations: buggy files and methods. (leave it empty if you don't have enough information)"
            )
            if isinstance(common.ACTOR_MODEL, ollama.OllamaModel):
                # llama models tend to always output search APIs and buggy locations.
                msg += "\n\nNOTE: If you have already identified the bug locations, do not make any search API calls."
            actor_msg_thread.add_user(msg)
            print_acr(
                msg,
                f"context retrieval round {round_no}",
                print_callback=print_callback,
            )

    round_no += 1
    # TODO: We need to differentiate situations where contexts collected are enough versus conv rounds limit reached
    intent = FunctionCallIntent("write_patch", {}, None)
    api_manager.start_new_tool_call_layer()
    api_manager.dispatch_intent(intent, actor_msg_thread, print_callback=print_callback)
    logger.info(f"Invoked {intent.func_name}.")

    logger.info("Ending workflow.")
    conversation_file = pjoin(output_dir, f"conversation_round_{round_no}.json")
    actor_msg_thread.save_to_file(conversation_file)

    return True


def search_for_bug_location(
    api_manager: ProjectApiManager,
    msg_thread: MessageThread,
    bug_location: dict[str, str],
) -> tuple[str, str, bool]:
    found = False

    file_name = bug_location.get("file")
    method_name = bug_location.get("method")
    class_name = bug_location.get("class")

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


def get_patch_explanation(fix_patch:str, api_manager: ProjectApiManager) -> dict:    
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
    # ZZ: Get explanation for critic model first 
    print_banner("Collecting explanations for the issue")
    exp_prompt, explanation_result = get_patch_explanation(fix_patch, api_manager)
    critic_msg_thread = MessageThread()
    critic_msg_thread.add_user(exp_prompt)
    critic_msg_thread.add_model(explanation_result, tools=[])
    
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
            output_dir, actor_msg_thread, critic_msg_thread, api_manager, print_callback=print_callback
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
