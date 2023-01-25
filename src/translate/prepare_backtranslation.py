#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 02.11.21
@author: gabriel.kressin@dfki.de
@description: prepares translated .jsonl file for backtranslation
"""

import argparse
import json
import logging


def change_keys(logger, input_file, output_file, overwrite):

    try:
        with open(output_file, "x", encoding="utf-8"):
            logger.info(f"Created file {output_file}")
    except FileExistsError:
        if overwrite:
            logger.warning(f"Overwriting existing file: {output_file}")
        else:
            logger.critical(
                f"Output file {output_file} already exists" + "- you may use --overwrite"
            )
            raise Exception("FileAlreadyExists")

    with open(input_file, "r", encoding="utf-8") as f_in, open(
        output_file, "w", encoding="utf-8"
    ) as f_out:

        for line in f_in:
            example = json.loads(line)
            example["tokens_original"] = example["tokens"]
            example["entities_original"] = example["entities"]
            example["language_original"] = example["language"]

            example["tokens"] = example.pop("tokens_translated")
            example["entities"] = example.pop("entities_translated")
            example["language"] = example.pop("language_translated")

            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")

    return True


def main():
    log_choices = ["debug", "info", "warning", "error", "critical"]

    # parse args
    parser = argparse.ArgumentParser(
        description="Script to prepare translated dataset for backtranslation,\n"
    )
    # Needed
    parser.add_argument("input_file", type=str, help="Path to translated dataset")
    parser.add_argument("output_file", type=str, help="Path to output file")
    # Optional
    parser.add_argument("--log_file", type=str, help="/path/to/log/file")
    parser.add_argument(
        "--log_level",
        default="warning",
        choices=log_choices,
        help="Adjust which kind of logs are saved/printed",
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help="Whether  output_file is overwritten if existing",
    )
    parser.add_argument("-v", action="store_true", help="Logs on console")

    args = parser.parse_args()

    # init logger
    logger_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    idx_choice = log_choices.index(args.log_level)
    log_level = logger_levels[idx_choice]

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    if args.v:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if args.log_file is not None:
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # call script
    return change_keys(
        logger=logger,
        input_file=args.input_file,
        output_file=args.output_file,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
