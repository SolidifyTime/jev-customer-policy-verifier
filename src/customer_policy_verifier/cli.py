"""One command dispatcher for dataset preparation, generation, review and training."""
from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "data": {
        "prepare": "data.prepare_data",
        "archive": "data.prepare_full_data",
        "build": "data.build_training_data",
    },
    "generate": {
        "pilot": "generation.pilot",
        "batch": "generation.batch_experiment",
        "expand": "generation.expand_scenes",
        "bulk": "generation.bulk_generate",
    },
    "review": {
        "report": "review.report_batch",
        "prepare": "review.prepare_manual_review",
        "import": "review.import_manual_review",
        "export": "review.export_candidates",
    },
}


def help_text() -> str:
    lines = ["Jev Customer Policy Verifier", "", "Usage: python scripts/cpv.py <group> <command> [options]", ""]
    for group, commands in COMMANDS.items():
        lines.append(f"  {group:9} " + " | ".join(commands))
    lines.extend(["  train     env | check | download | audit | smoke | train | evaluate | package",
                  "", "Append --help to a command for its options.",
                  "Example: python scripts/cpv.py train check"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(help_text())
        return 0
    if argv[0] == "train":
        if len(argv) > 1 and argv[1] == "env":
            module_name, remaining = "training.check_environment", argv[2:]
        else:
            module_name, remaining = "training.train_jev", argv[1:]
    else:
        if len(argv) < 2 or argv[0] not in COMMANDS or argv[1] not in COMMANDS[argv[0]]:
            raise SystemExit("Unknown command.\n\n" + help_text())
        module_name, remaining = COMMANDS[argv[0]][argv[1]], argv[2:]
    module = importlib.import_module("customer_policy_verifier." + module_name)
    previous = sys.argv
    try:
        sys.argv = ["cpv " + " ".join(argv[:1 if argv[0] == "train" else 2]), *remaining]
        module.main()
        return 0
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
