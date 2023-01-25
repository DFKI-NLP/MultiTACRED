#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 06.10.21
@author: leonhard.hennig@dfki.de, gabriel.kressin@dfki.de

For convenience it is best to create a .env file of following
format:

INPUT_FILE='/path/to/dataset.json'
OUTPUT_FILE='output.jsonl'
API_KEY='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:xx'
API_ADDRESS='https://api.deepl.com/v2/translate'
"""

import argparse

import requests
from dotenv import dotenv_values
from utils import LOG_CHOICES, TOKENIZER_CHOICES, init_logger, init_tokenizer, translate_dataset


def list_languages(api_key, api_address):
    api_address = api_address.replace("translate", "languages")
    data = {"auth_key": api_key, "type": "source"}
    response = requests.post(api_address, data=data)
    results = response.json()

    print("### Available source languages")
    for lang_dic in results:
        print(f'{lang_dic["name"]} ({lang_dic["language"]})')

    data["type"] = "target"
    response = requests.post(api_address, data=data)
    results = response.json()

    print("### Available target languages")
    for lang_dic in results:
        print(f'{lang_dic["name"]} ({lang_dic["language"]})')


class TranslateDeepl:
    def __init__(self, api_key, api_address):
        self.api_key = api_key
        self.api_address = api_address
        self.key = "text"

    def translate(self, texts, source_language, target_language):
        """
        Sends strings in texts to DeepL for translation

        Known Caveats:
        - Some translations into German combine multiple tokens into a compound. If Head or Tail in the
        English original encompassed only a single token, the resulting German Head/Tail may be different, e.g.
        '<H>ETA</H> members' gets translated to '<H>ETA-Mitglieder</H>', thus somewhat changing the semantics of
        the instance.
        - DeepL strips leading/trailing quote characters, which sometimes leads to 'invalid' quotations
        where only the text-internal quote is transferred to the target language,
        e.g. '``bla bla bla'', X said' gets translated to 'bla bla bla\", sagte X'.
        We currently ignore this, because it doesn't change the semantics of the instance much.

        - DeepL also converts all paired single quotes to the '"' character.

        - Ex. In e779865fb91d8e964a08, "US-Dollar" is translated to '$', but the tail tags are wrongly assigned to 'in'

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
            data = {
                "auth_key": self.api_key,
                "text": texts,
                "split_sentences": "1",  # enabled 25.03.2022 Leo - some TACRED examples work only with
                # sent splitting, e.g. id=61b3a65fb9bb1419f614. Doesn't seem to affect other examples
                "tag_handling": "xml",
                "outline_detection": "0",
                "source_lang": source_language,
                "target_lang": target_language,
            }

            response = requests.post(self.api_address, data=data)

            scode = response.status_code

            if not scode == 200:
                # Error Handling

                message = f"Error {scode}: "
                if scode == 400:
                    message += (
                        "Bad request. Please check error message"
                        " and your parameters. Is your target"
                        " language valid?"
                    )
                elif scode == 403:
                    message += (
                        "Forbidden. The access to the requested"
                        " resource is denied, because of insufficient access"
                        " rights. Check your api_key."
                    )
                elif scode == 429 or scode == 529:
                    message += "Too many requests. Please wait and" " resend your request."
                elif scode == 456:
                    message += "Quota exceeded. The character limit has" " been reached."
                else:
                    message += (
                        "Please refer to "
                        "https://www.deepl.com/docs-api/"
                        "accessing-the-api/error-handling/"
                    )

                message += f"\nHTTP message: {response.reason}"
                raise Exception(message)

            result = response.json()["translations"]

            return True, result, self.key, None
        except Exception as err:
            return False, None, self.key, err


def main():
    # parse args
    parser = argparse.ArgumentParser(
        description="Script to translate a dataset in .json format,\n"
        "arguments will override .env file."
    )
    parser.add_argument("--api_key", type=str, help="API Key for deepl API")
    parser.add_argument("--api_address", type=str, help="URL for deepl API")
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
        help="Maximum of translated characters in a run",
    )
    parser.add_argument("-o", "--output_file", type=str, help="Path to output file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether existing outputfile is overwritten"
    )
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

    if args.show_languages:
        return list_languages(api_key, api_address)

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
    translateDeepl = TranslateDeepl(api_key, api_address)

    return translate_dataset(
        logger=logger,
        translation_function=translateDeepl.translate,
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
