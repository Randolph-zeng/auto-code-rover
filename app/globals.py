"""
Values of global configuration variables.
"""

# Overall output directory for results
output_dir: str = ""

# upper bound of the number of conversation rounds for the agent
conv_round_limit: int = 15

# whether to perform sbfl
enable_sbfl: bool = False

# whether to perform layered search
enable_layered: bool = True

# whether to perform our own validation
enable_validation: bool = False

# whether to do angelic debugging
enable_angelic: bool = False

# whether to do perfect angelic debugging
enable_perfect_angelic: bool = False


# A special mode to only save SBFL result and exit
only_save_sbfl_result: bool = False

# timeout for test cmd execution, currently set to 5 min
test_exec_timeout: int = 300

# A special mode to only collect fix locations and exit
disable_patch_generation: bool = False

# Used with disable_patch_generation - constrains or extends the amount of context retrieval rounds
context_generation_limit: int = -1

# ZZ: The number of possible outcomes to sample at each step for rejection sampling.
rejection_sampling_k: int = -1

# ZZ: The budget control parameters for rejection sampling rounds and overall api request numbers
rejection_sampling_rounds: int = 3
total_request_num: int = 100
