"""Compatibility wrapper for the moved modify-codegen SFT export script."""

from scripts.analysis.modify_codegen.export_modify_codegen_sft_dataset import *  # noqa: F401,F403
from scripts.analysis.modify_codegen.export_modify_codegen_sft_dataset import main


if __name__ == "__main__":
    main()
