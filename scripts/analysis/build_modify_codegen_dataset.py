"""Compatibility wrapper for the moved modify-codegen dataset builder."""

from scripts.analysis.modify_codegen.build_modify_codegen_dataset import *  # noqa: F401,F403
from scripts.analysis.modify_codegen.build_modify_codegen_dataset import main


if __name__ == "__main__":
    main()
