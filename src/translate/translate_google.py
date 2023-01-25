#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 12.10.21
@author: gabriel.kressin@dfki.de

In order for the script to work you need to set up the google
account properly: https://cloud.google.com/translate/docs/setup
Create private key and copy path into "PRIVATE_KEY" variable
or use the option `--private_key=/path/to/private_key` when calling
the script.

For convenience it is best to create a .env file of following
format:

INPUT_FILE='/path/to/dataset.json'
OUTPUT_FILE='output.jsonl'
PRIVATE_KEY='/path/to/private_key'
"""

import argparse

from dotenv import dotenv_values
from google.cloud import translate_v2 as gtranslate
from utils import LOG_CHOICES, TOKENIZER_CHOICES, init_logger, init_tokenizer, translate_dataset


def list_languages(private_key):
    """Lists all available languages."""

    translate_client = gtranslate.Client.from_service_account_json(private_key)

    results = translate_client.get_languages()

    for lang_dic in results:
        print(f'{lang_dic["name"]} ({lang_dic["language"]})')

    return 0


class TranslateGoogle:
    def __init__(self, private_key):
        self.translation_client = gtranslate.client.Client.from_service_account_json(private_key)
        self.key = "translatedText"

    def translate(self, texts, source_language, target_language):
        """
        Sends strings in texts to google for translation

        Parameters
        ----------
        texts:
            list of strings to be translated
        source_language:
            string identifier for source text
        target_language:
            string identifier for translation target

        Returns
        -------
        tuple: (bool_success, list_dict_translations, key, error)
            bool_success:
                whether translation was successfull
            list_dict_translations:
                list of dicts with translated texts at key 'translations'
            key:
                string with key to access text in dictionaries
            error: Exception
                Exception object
        """
        try:
            response = self.translation_client.translate(
                texts,
                target_language=target_language,
                format_="html",
                source_language=source_language,
                model="nmt",
            )
            return True, response, self.key, None
        except Exception as err:
            return False, None, self.key, err


def main():
    # parse args
    parser = argparse.ArgumentParser(
        description="Script to translate a dataset in .json format,\n"
        "cli parameters will override .env file."
    )
    parser.add_argument("-i", "--input_file", type=str, help="Path to dataset.json")
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
    parser.add_argument("-o", "--output_file", type=str, help="Path to output file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether existing outputfile is overwritten"
    )
    parser.add_argument("--private_key", type=str, help="Private Key for google")
    parser.add_argument("--show_languages", action="store_true")
    parser.add_argument("-s", "--source_lang", type=str, default="EN")
    parser.add_argument(
        "-t", "--target_lang", type=str, default="DE", help="Supported code for language"
    )
    parser.add_argument("-T", "--tokenizer", default="split", choices=TOKENIZER_CHOICES)
    parser.add_argument("-v", action="store_true", help="Logs on console")

    args = parser.parse_args()
    print(args)
    # init logger
    logger = init_logger(__name__, args.v, args.log_level, args.log_file)

    # load dotenv
    config = dotenv_values()

    # Check if arguments are in .env or in input
    if args.private_key is None:
        if "PRIVATE_KEY" not in config.keys():
            message = "No Private key. Please specify Private key in .env or as arg."
            logger.critical(message)
            raise Exception(message)
        private_key = config["PRIVATE_KEY"]
    else:
        private_key = args.private_key

    if args.show_languages:
        return list_languages(private_key)

    if args.input_file is None:
        if "INPUT_FILE" not in config.keys():
            message = "No input file. Please specify input file in .env or as arg."
            logger.critical(message)
            raise Exception(message)
        input_file = config["INPUT_FILE"]
    else:
        input_file = args.input_file
    if args.output_file is None:
        if "OUTPUT_FILE" not in config.keys():
            message = "No output file. Please specify output file in .env or as arg."
            logger.critical(message)
            raise Exception(message)
        output_file = config["OUTPUT_FILE"]
    else:
        output_file = args.output_file

    # Init tokenizer
    tokenizer = init_tokenizer(logger, args.tokenizer)

    # Init function that sends data to google and returns result
    translateGoogle = TranslateGoogle(private_key=private_key)

    return translate_dataset(
        logger=logger,
        translation_function=translateGoogle.translate,
        input_file=input_file,
        output_file=output_file,
        tokenizer=tokenizer,
        source_language=args.source_lang,
        target_language=args.target_lang,
        overwrite_translated=args.overwrite,
        max_characters=args.max_characters,
    )


if __name__ == "__main__":
    main()
