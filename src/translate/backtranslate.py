#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 03.11.21
@author: gabriel.kressin@dfki.de
@description: backtranslate dataset.json file to original language
"""

import argparse
import json
import os

import postpare_backtranslation
import prepare_backtranslation
from dotenv import dotenv_values
from utils import LOG_CHOICES, TOKENIZER_CHOICES, init_logger, init_tokenizer, translate_dataset


def backtranslate(
    logger,
    input_file,
    output_file,
    service,  # 'google' or 'deepl'
    api_key,  # arguments from translation scripts
    api_address,
    private_key,
    tokenizer,
    overwrite_translated,
    max_characters,
):
    # 0. temporary files
    temp_file = f"{input_file}.tmp"
    if os.path.isfile(temp_file):
        logger.info(f"Overwriting {temp_file}")

    temp_file_translated = f"{output_file}.tmp"
    if os.path.isfile(temp_file_translated):
        if overwrite_translated:
            logger.warning("Overwriting previous backtranslation" + f" in {temp_file_translated}")
        else:
            logger.info("Appending to previous backtranslation" + f" in {temp_file_translated}")

    # 1. prepare json key fields
    prepare_backtranslation.change_keys(
        logger=logger, input_file=input_file, output_file=temp_file, overwrite=True
    )

    # 2. get both language identities
    with open(temp_file, "r", encoding="utf-8") as f_temp:
        line = f_temp.readline()
        example = json.loads(line)
        source_language = example["language"]
        target_language = example["language_original"]

    # 3. translate
    if service == "deepl":
        import translate_deepl

        translateDeepl = translate_deepl.TranslateDeepl(api_key, api_address)
        translate_dataset(
            logger=logger,
            translation_function=translateDeepl.translate,
            input_file=temp_file,
            output_file=temp_file_translated,
            tokenizer=tokenizer,
            source_language=source_language,
            target_language=target_language,
            overwrite_translated=overwrite_translated,
            max_characters=max_characters,
        )
    if service == "google":
        import translate_google

        translateGoogle = translate_google.TranslateGoogle(private_key)
        translate_dataset(
            logger=logger,
            translation_function=translateGoogle.translate,
            input_file=temp_file,
            output_file=temp_file_translated,
            tokenizer=tokenizer,
            source_language=source_language,
            target_language=target_language,
            overwrite_translated=overwrite_translated,
            max_characters=max_characters,
        )

    logger.info(
        f"Remember to check {temp_file_translated}.manual"
        + " for failed backtranslations. Add them manually to"
        + f" {temp_file} and call postpare_backtranslation on it."
    )

    # 4. bring fields into right format again
    postpare_backtranslation.change_keys(
        logger=logger, input_file=temp_file_translated, output_file=output_file, overwrite=True
    )

    return 0


def main():
    services = ["deepl", "google"]

    # parse args
    parser = argparse.ArgumentParser(
        description="Script to translate a dataset in .json format,\n"
        "arguments will override .env file."
    )
    # Required
    parser.add_argument("input_file", type=str, help="Path to translated dataset")
    parser.add_argument("output_file", type=str, help="Path to output file")
    parser.add_argument("service", choices=services, help="Tranlation Service to use")
    # Optional
    parser.add_argument("--api_key", type=str, help="API Key for deepl API")
    parser.add_argument("--api_address", type=str, help="URL for deepl API")
    parser.add_argument("--log_file", type=str, help="/path/to/log/file")
    parser.add_argument(
        "--log_level",
        default="info",
        choices=LOG_CHOICES,
        help="Adjust which kind of logs are saved/printed",
    )
    parser.add_argument(
        "-m",
        "--max_characters",
        type=int,
        default=-1,
        help="Maximum of translated characters in a run. Negative numbers for infinite.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether existing temporary backtranslation is overwritten",
    )
    parser.add_argument("--private_key", type=str, help="Private Key for google")
    parser.add_argument("-T", "--tokenizer", default="split", choices=TOKENIZER_CHOICES)
    parser.add_argument("-v", action="store_true", help="Logs on console")

    args = parser.parse_args()

    # init logger
    logger = init_logger(__name__, args.v, args.log_level, args.log_file)

    config = dotenv_values()

    api_key = None
    api_address = None
    private_key = None
    if args.service == "deepl":
        # Check if deepl arguments are in .env or in input
        if args.api_key is None:
            if "API_KEY" not in config.keys():
                message = "No API key. Please specify API key in .env or as arg."
                logger.critical(message)
                raise Exception(message)
            api_key = config["API_KEY"]
        else:
            api_key = args.api_key
        if args.api_address is None:
            if "API_ADDRESS" not in config.keys():
                message = "No API ADDRESS. Please specify API ADDRESS in .env or as arg."
                logger.critical(message)
                raise Exception(message)
            api_address = config["API_ADDRESS"]
        else:
            api_address = args.api_address
    elif args.service == "google":
        # Check if google arguments are in .env or in input
        if args.private_key is None:
            if "PRIVATE_KEY" not in config.keys():
                message = "No Private key. Please specify Private key in .env or as arg."
                logger.critical(message)
                raise Exception(message)
            private_key = config["PRIVATE_KEY"]
        else:
            private_key = args.private_key

    # Init tokenizer
    tokenizer = init_tokenizer(logger, args.tokenizer)

    return backtranslate(
        logger=logger,
        input_file=args.input_file,
        output_file=args.output_file,
        service=args.service,
        api_key=api_key,
        api_address=api_address,
        private_key=private_key,
        tokenizer=tokenizer,
        overwrite_translated=args.overwrite,
        max_characters=args.max_characters,
    )


if __name__ == "__main__":
    main()
