##################################################################################################
#####    THIS FILE EXECUTES THE EXPERIMENTS AND ANALYSES CHOSEN IN THE CONFIG FILE    #####
#####              -----------------------------------------------------------               #####
##################################################################################################

from run_pipeline.config.registry import FUNCTION_REGISTRY
from run_pipeline.config.run_plan import TOUCHPOINT_PLAN
from run_pipeline.config.runner import run_touchpoint_plan

# For local runs, keep require_acceptance=True: the runner prints the selected
# program and asks before doing damage. If you want to run without asking for acceptance
# then require_acceptance=False,
# or assume_yes=True.

def main() -> None:
    run_touchpoint_plan(
        TOUCHPOINT_PLAN,
        FUNCTION_REGISTRY,
        run_experiments=False,
        run_analyses=True,
        require_acceptance=True,
    )


if __name__ == "__main__":
    main()

