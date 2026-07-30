"""Write the compact record and Markdown report for the Multi30k factorial screen."""

from __future__ import annotations

import argparse

from ana.factorial import STUDY_ID, write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=f"runs/{STUDY_ID}")
    parser.add_argument("--json", default=f"results/{STUDY_ID}.json")
    parser.add_argument("--markdown", default=f"results/{STUDY_ID}.md")
    args = parser.parse_args()

    write_artifacts(args.runs, args.json, args.markdown)
    print(f"wrote {args.json}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
