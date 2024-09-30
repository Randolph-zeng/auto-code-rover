import argparse
import json
import logging
import os
import shutil
import pandas as pd
from git import Repo
from multiprocessing import Pool
from os.path import join as pjoin
from typing import Dict, Optional


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_setup")


def clone_repo(repo_name: str, path: str, token: str = None, max_retry: int = 5) -> bool:
    """
    Wrapper for cloning repo from swe-bench organization

    Args:
        repo_name (str): Name of repo to clone
        path (str): Path to clone repo to
        token (str): GitHub token to use for cloning
    Returns:
        success (bool): True if repo cloned successfully, False otherwise
    """
    # ZZ: Keep retrying until success or max retry reached 
    retry_count = 0
    while retry_count < max_retry:
        try:
            if token is None:
                token = os.environ.get("GITHUB_TOKEN", "git")
            repo_org, repo_str = repo_name.split("/")
            repo_url = (
                f"https://{token}@github.com/{repo_org}/"
                + repo_str
                + ".git"
            )
            Repo.clone_from(repo_url, path)
            return True
        except Exception as e:
            print(e)
            retry_count += 1
    return False


def create_fresh_dir(dir_name: str):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def create_if_not_exist(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def setup_one_repo_version(
    repo_full: str, repo_path: str, version: str, env_name: str, task: Dict
):
    """
    Main entry for setting up one repo+version combination.
    Put all logic in one function so it's easy to parallelize.
    Args:
        repo_full: full name of the repo, in the form "user/repo".
        repo_path: the cloned repo path on disk.
        version: version of the repo.
        env_name: name of the conda environment to create.
        task: Dict containing task instance.
    """
    logger.info(
        f"[{env_name}] ======= Start setting up for {repo_full} {version} ======="
    )
    clone_repo(repo_full, repo_path)
    logger.info(f"[{env_name}] Cloned {repo_full} to {repo_path}")
    

def get_pr_link_for_task(task: Dict):
    task_id = task["instance_id"]
    repo_long, _, pr_id = task_id.rpartition("-")  # split on the last "-"
    owner, repo_name = repo_long.split("__")
    pr_link = f"https://github.com/{owner}/{repo_name}/pull/{pr_id}"
    return pr_link


def load_task_instances(swe_bench_tasks: str):
    # for parquet version
    df = pd.read_parquet(swe_bench_tasks, engine="pyarrow")
    tasks = json.loads(df.to_json(orient="records"))
    # now form a link to PR for each meta data entry
    for t in tasks:
        pr_link = get_pr_link_for_task(t)
        t["pr_link"] = pr_link
    # fields that are supposed to be list, are encoded as string in parquet
    # fix them here
    for t in tasks:
        fail_to_pass = t["FAIL_TO_PASS"]
        t["FAIL_TO_PASS"] = json.loads(fail_to_pass)
        pass_to_pass = t["PASS_TO_PASS"]
        t["PASS_TO_PASS"] = json.loads(pass_to_pass)
    return tasks
    # this is for the json version, which is deprecated
    # if not os.path.exists(swe_bench_tasks):
    #     raise ValueError("--swe_bench_tasks does not exist")
    # tasks = json.load(open(os.path.abspath(swe_bench_tasks)))
    # if not isinstance(tasks, list):
    #     raise ValueError(f"{swe_bench_tasks} must contain an array of tasks")
    # return tasks


def save_setup_json_files(result_dir: str, setup_map: Dict, tasks_map: Dict):
    """
    Dump maps containing setup information to disk, so other clients can
    use them to find locations of the setup.
    """
    setup_map_path = pjoin(result_dir, "setup_map.json")
    tasks_map_path = pjoin(result_dir, "tasks_map.json")
    with open(setup_map_path, "w") as f:
        json.dump(setup_map, f)
    with open(tasks_map_path, "w") as f:
        json.dump(tasks_map, f)

    print("Done with setup.")
    print(f"setup_map is saved to {setup_map_path}")
    print(f"tasks_map is saved to {tasks_map_path}")


def main(
    swe_bench_tasks: str,
    log_dir: str,
    testbed: str,
    result_dir: str,
    num_processes: int = 1,
    subset_file: Optional[str] = None,
    only_dump_files: bool = False,
):
    """
    Runs set up for each repo/version combination.

    Args:
        swe_bench_tasks (str): Path to the SWE-bench tasks file.
        log_dir (str): Path to the directory where logs will be saved.
        testbed (str): Path to the directory where testbeds will be saved.
        result_dir (str): Path to the directory where results are stored.
        num_processes (int, optional): Number of processes to use.
        subset_file (str, optional): Path to a file indicating which subset to set up.
        only_dump_files (bool, optional): Only dump json files without performing the actual setup.
    """
    # since we are going to store testbed dirs for others to use, we should use absolute path
    testbed = os.path.abspath(testbed)
    # if we just want to dump files, do not touch log and testbed dirs
    if not only_dump_files:
        create_fresh_dir(log_dir)
        create_fresh_dir(testbed)
    create_fresh_dir(result_dir)
    tasks = load_task_instances(swe_bench_tasks)
    # ZZ: Note we only save info about tasks we really setup to save time and space
    # map instance_id to the actual task instance Dict
    full_tasks_map, filtered_tasks_map = {t['instance_id']: t for t in tasks}, {}
    # map instance_id to setup information
    setup_map = {}

    # sometimes we just want to setup a subset of instances for quick experiments
    selected_instances = []  # only used if there is a subset_file
    if subset_file is not None:
        with open(subset_file, "r") as f:
            selected_instances = [line.strip() for line in f.readlines()]

    # keep all information for setting up each entry
    setup_entries = []
    # iterates all tasks, decide which ones need setup,
    # decide the path for their testbed folder, and save this path to task_map
    for instance_id, task in full_tasks_map.items():
        if subset_file is not None and instance_id not in selected_instances:
            continue
        setup_map[instance_id] = {}
        filtered_tasks_map[instance_id] = full_tasks_map[instance_id]
        repo_full = task["repo"]  # "astropy/astropy"
        repo_short = instance_id.rsplit("-", 1)[0]  # "astropy"
        version = task["version"]  # "4.2"
        # name for both conda env and testbed folder
        env_name = f"setup_{repo_short}__{version}"
        repo_path = pjoin(testbed, repo_short, env_name)
        # keys in setup_map
        setup_map[instance_id]["repo_path"] = repo_path
        setup_map[instance_id]["env_name"] = env_name
        setup_map[instance_id]["pre_install"] = ''
        setup_map[instance_id]["install"] = ''
        setup_map[instance_id]["test_cmd"] = ''
        collected_entry_env_names = [e[3] for e in setup_entries]
        if env_name in collected_entry_env_names:
            # this repo+version combination has been recorded before
            continue
        # should really do setup
        setup_entries.append((repo_full, repo_path, version, env_name, task))

    setup_entries = sorted(setup_entries, key=lambda x: x[3])
    all_env_names = [e[3] for e in setup_entries]
    logger.info(f"env_name for all setup entries: {all_env_names}")

    # Now we have all the information for setting up each entry
    num_setup_entries = len(setup_entries)
    num_processes = min(num_processes, num_setup_entries)
    if num_setup_entries == 0:
        logger.info("No setup needed.")
        return

    logger.info("Starting parallel setup.")
    logger.info(f"\tNumber of setup tasks: {num_setup_entries}")
    logger.info(f"\tNumber of processes: {num_processes}")
    try:
        if num_processes == 1:
            for entry in setup_entries:
                setup_one_repo_version(*entry)
        else:
            # parallel
            pool = Pool(processes=num_processes)
            pool.starmap(setup_one_repo_version, setup_entries)
            pool.close()
            pool.join()
    finally:
        # Done with the actual work.
        save_setup_json_files(result_dir, setup_map, filtered_tasks_map)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(script_dir)
    # we always read from this file, so put this as a default instead of required
    default_tasks_file = pjoin(
        root_dir, "setup", "swe-bench-train.parquet"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir", type=str, help="Path to log directory", required=True
    )
    parser.add_argument(
        "--swe_bench_tasks",
        type=str,
        help="Path to SWE-bench task instances file",
        default=default_tasks_file,
    )
    parser.add_argument(
        "--testbed", type=str, help="Path to testbed directory", required=True
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        help="Directory to store the setup result maps",
        required=True,
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="(Optional) Number of processes to use.",
        default=1,
    )
    parser.add_argument(
        "--subset_file",
        type=str,
        help="(Optional) Path to a file containing a subset of instances to setup. Each line should contain one instace id to be set up.",
        default=None,
    )
    args = parser.parse_args()
    main(**vars(args))
