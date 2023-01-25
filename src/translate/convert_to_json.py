"""
Converts JSONL format from translation into the TACRED JSON format.

JSON output format:
{
 "id": "61b3a65fb9b7111c4ca4",
 "token": ["In", "1983", ",", "a", "year", "after", "the", "rally", ",", "Forsberg", "received", "the", "so-called", "``", "genius", "award", "''", "from", "the", "John", "D.", "and", "Catherine", "T.", "MacArthur", "Foundation", "."],
 "relation": "no_relation",
 "subj_start": 9,
 "subj_end": 9,
 "obj_start": 19,
 "obj_end": 20,
 "subj_type": "PERSON",
 "obj_type": "PERSON"
}
"""

import argparse
import json
import os

from tqdm import tqdm


class DatasetConverter:
    def __init__(self, dataset_dir, output_dir, language):

        self.input_train_file = os.path.join(dataset_dir, f"tacred_train_{language}.jsonl")
        self.input_test_file = os.path.join(dataset_dir, f"tacred_test_{language}.jsonl")
        self.input_dev_file = os.path.join(dataset_dir, f"tacred_dev_{language}.jsonl")
        self.input_backtranslation_file = os.path.join(
            dataset_dir, f"tacred_test_en_{language}_bt.jsonl"
        )

        self.output_dir = output_dir

        assert os.path.exists(
            self.input_train_file
        ), f"Train file not found: {self.input_train_file}"
        self.output_train_file = os.path.join(output_dir, f"train_{language}.json")

        assert os.path.exists(self.input_test_file), f"Test file not found: {self.input_test_file}"
        self.output_test_file = os.path.join(output_dir, f"test_{language}.json")

        assert os.path.exists(self.input_dev_file), f"Dev file not found: {self.input_dev_file}"
        self.output_dev_file = os.path.join(output_dir, f"dev_{language}.json")

        if not os.path.exists(self.input_backtranslation_file):
            print(f"Test Backtranslation file not found: {self.input_backtranslation_file}")
            self.output_test_bt_file = None
        else:
            self.output_test_bt_file = os.path.join(output_dir, f"test_en_{language}_bt.json")

        self.glove_mapping = {
            "(": "-LRB-",
            ")": "-RRB-",
            "[": "-LSB-",
            "]": "-RSB-",
            "{": "-LCB-",
            "}": "-RCB-",
        }

    def run(self):
        print("Converting dataset to TACRED json")
        os.makedirs(self.output_dir, exist_ok=True)
        self._run_normally()

    def _run_normally(self):
        # Convert the dev and test set
        self._convert_tacred_format_file(self.input_test_file, self.output_test_file)
        self._convert_tacred_format_file(self.input_dev_file, self.output_dev_file)
        self._convert_tacred_format_file(self.input_train_file, self.output_train_file)
        if self.output_test_bt_file:
            self._convert_tacred_format_file(
                self.input_backtranslation_file, self.output_test_bt_file, is_backtranslation=True
            )

    def _convert_tacred_format_file(self, input_file, output_file, is_backtranslation=False):
        with open(output_file, "w") as output_file:
            data = []
            for example in self._read_jsonl_file(
                input_file, is_backtranslation=is_backtranslation
            ):
                data.append(example)
            output_file.write(json.dumps(data, ensure_ascii=False))

    def _read_jsonl_file(self, input_file, is_backtranslation=False):
        with open(input_file, "r") as input_file:
            for line in tqdm(input_file):
                input_example = json.loads(line)
                key_suffix = "translated" if not is_backtranslation else "backtranslated"
                tokens = input_example[f"tokens_{key_suffix}"]

                tokens = self.renormalize_glove_tokens(tokens)

                output_example = {
                    "id": input_example["id"],
                    "token": tokens,
                    "relation": input_example["label"],
                    "subj_start": input_example[f"entities_{key_suffix}"][0][0],
                    # -1 because TACRED format uses inclusive end index
                    "subj_end": input_example[f"entities_{key_suffix}"][0][1] - 1,
                    "obj_start": input_example[f"entities_{key_suffix}"][1][0],
                    # -1 because TACRED format uses inclusive end index
                    "obj_end": input_example[f"entities_{key_suffix}"][1][1] - 1,
                    "subj_type": input_example["type"][0],
                    "obj_type": input_example["type"][1],
                }

                yield output_example

    def renormalize_glove_tokens(self, tokens):
        return [
            self.glove_mapping[token] if token in self.glove_mapping else token for token in tokens
        ]


def main(args):
    assert os.path.exists(args.dataset_dir), "Input directory does not exist"
    converter = DatasetConverter(args.dataset_dir, args.output_dir, args.language)
    converter.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, help="The directory containing the translated files"
    )
    parser.add_argument("--output_dir", type=str, help="An output directory of json files")
    parser.add_argument(
        "--language", type=str, help="The language of the translations, as a 2-letter ISO code"
    )

    args = parser.parse_args()
    print(args)
    main(args)
