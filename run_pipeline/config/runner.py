from __future__ import annotations

from typing import Any, Callable


Step = dict[str, Any]
Touchpoint = dict[str, Any]


def selected_touchpoints(touchpoint_plan: list[Touchpoint]) -> list[Touchpoint]:
    return [
        tp
        for tp in touchpoint_plan
        if tp.get("run_experiments", False) or tp.get("run_analyses", False)
    ]


def flatten_steps(
    touchpoint_plan: list[Touchpoint],
    *,
    run_experiments: bool = True,
    run_analyses: bool = True,
) -> list[Step]:
    steps: list[Step] = []

    for tp in selected_touchpoints(touchpoint_plan):
        if run_experiments and tp.get("run_experiments", False):
            for step in tp.get("experiments", []):
                steps.append(
                    {
                        **step,
                        "touchpoint": tp["key"],
                        "label": tp["label"],
                    }
                )

        if run_analyses and tp.get("run_analyses", False):
            for step in tp.get("analyses", []):
                steps.append(
                    {
                        **step,
                        "touchpoint": tp["key"],
                        "label": tp["label"],
                    }
                )

    return steps


def validate_steps(
    steps: list[Step],
    registry: dict[str, Callable[..., Any]],
) -> None:
    missing = sorted({step["name"] for step in steps if step["name"] not in registry})

    if missing:
        raise KeyError(
            "Run plan references unknown function name(s):\n"
            f"  missing = {missing}\n\n"
            f"Known functions:\n"
            f"  {sorted(registry)}"
        )


def print_program(steps: list[Step]) -> None:
    """Print the exact execution program, in order."""

    if not steps:
        print("[runner] Nothing selected.")
        return

    print("\n" + "=" * 88)
    print("[runner] Execution program")
    print("=" * 88)

    current_touchpoint = None

    for i, step in enumerate(steps, start=1):
        touchpoint = step["touchpoint"]

        if touchpoint != current_touchpoint:
            current_touchpoint = touchpoint
            print(f"\n[{touchpoint}]")
            print(f"  {step['label']}")

        kind = step["kind"]
        name = step["name"]
        params = step.get("params", {})

        param_keys = ", ".join(sorted(params)) if params else "<none>"

        print(f"  {i:02d}. {kind:<10} {name}")
        print(f"      params: {param_keys}")

    print("\n" + "=" * 88)


def ask_for_acceptance(
    *,
    prompt: str = "Run this program? [y/N]: ",
    assume_yes: bool = False,
) -> bool:
    """Return True only if user explicitly accepts."""

    if assume_yes:
        print("[runner] assume_yes=True, skipping confirmation.")
        return True

    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def run_steps(
    steps: list[Step],
    registry: dict[str, Callable[..., Any]],
) -> None:
    if not steps:
        print("[runner] Nothing to run.")
        return

    for i, step in enumerate(steps, start=1):
        name = step["name"]
        kind = step["kind"]
        params = step.get("params", {})

        print(f"\n[runner] {i}/{len(steps)} {kind}: {name}")
        print(f"[runner] touchpoint: {step['touchpoint']}")
        print(f"[runner] params: {sorted(params)}")

        try:
            registry[name](**params)
        except Exception as exc:
            raise RuntimeError(
                f"Failed while running step: {name}\n"
                f"Kind: {kind}\n"
                f"Touchpoint: {step['touchpoint']}\n"
                f"Param keys: {sorted(params)}"
            ) from exc
        

def warn_if_globally_blocked(
    touchpoint_plan: list[Touchpoint],
    *,
    run_experiments: bool,
    run_analyses: bool,
) -> None:
    selected_experiments = [
        tp["key"]
        for tp in touchpoint_plan
        if tp.get("run_experiments", False)
    ]

    selected_analyses = [
        tp["key"]
        for tp in touchpoint_plan
        if tp.get("run_analyses", False)
    ]

    if selected_experiments and not run_experiments:
        print("\n[runner] WARNING: experiment touchpoints are selected, but run_experiments=False.")
        print("[runner] Blocked experiment touchpoints:")
        for key in selected_experiments:
            print(f"  - {key}")

    if selected_analyses and not run_analyses:
        print("\n[runner] WARNING: analysis touchpoints are selected, but run_analyses=False.")
        print("[runner] Blocked analysis touchpoints:")
        for key in selected_analyses:
            print(f"  - {key}")


def run_touchpoint_plan(
    touchpoint_plan: list[Touchpoint],
    registry: dict[str, Callable[..., Any]],
    *,
    run_experiments: bool = True,
    run_analyses: bool = True,
    require_acceptance: bool = True,
    assume_yes: bool = False,
) -> None:
    warn_if_globally_blocked(
        touchpoint_plan,
        run_experiments=run_experiments,
        run_analyses=run_analyses,
    )

    steps = flatten_steps(
        touchpoint_plan,
        run_experiments=run_experiments,
        run_analyses=run_analyses,
    )

    validate_steps(steps, registry)
    print_program(steps)

    if not steps:
        return

    if require_acceptance:
        accepted = ask_for_acceptance(assume_yes=assume_yes)

        if not accepted:
            print("[runner] Aborted. Nothing was run.")
            return

    run_steps(steps, registry)

