"""Keep report writes separate from immutable experiment source directories."""
from pathlib import Path


def validate_report_output(output, *source_directories):
    output = Path(output).resolve()
    for source in source_directories:
        source = Path(source).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"Report output must not overlap a source directory: {output} and {source}")
    # Resolving the output argument alone does not catch an existing artifacts/
    # symlink that redirects later writes back into a run or another directory.
    if output.is_dir() and any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("Report output must not contain symlinks that redirect artifact writes")
