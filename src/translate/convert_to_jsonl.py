"""
Converts TACRED into a jsonl format.

JSONL output format:
{
 "id": "61b3a65fb9b7111c4ca4",
 "tokens": ["In", "1983", ",", "a", "year", "after", "the", "rally", ",", "Forsberg", "received", "the", "so-called", "``", "genius", "award", "''", "from", "the", "John", "D.", "and", "Catherine", "T.", "MacArthur", "Foundation", "."],
 "label": "no_relation",
 "entities": [[9, 10], [19, 21]],
 "grammar": ["SUBJ", "OBJ"],
 "type": ["PERSON", "PERSON"]
}
"""

import argparse
import json
import os


class DatasetConverter:
    def __init__(self, dataset_dir, output_dir, language=None):

        self.input_train_file = os.path.join(dataset_dir, "train.json" if language is None else f"train_{language}.json")
        self.input_test_file = os.path.join(dataset_dir, "test.json" if language is None else f"test_{language}.json")
        self.input_dev_file = os.path.join(dataset_dir, "dev.json" if language is None else f"dev_{language}.json")
        self.input_test_backtranslation_file = None
        if language is not None:
            self.input_test_backtranslation_file = os.path.join(dataset_dir, f"test_en_{language}_bt.json")
        self.output_dir = output_dir

        assert os.path.exists(self.input_train_file), "Train file not found: {}".format(
            self.input_train_file
        )
        self.output_train_file = os.path.join(output_dir, "train.jsonl" if language is None else f"train_{language}.jsonl")

        assert os.path.exists(self.input_test_file), "Test file not found: {}".format(
            self.input_test_file
        )
        self.output_test_file = os.path.join(output_dir, "test.jsonl" if language is None else f"test_{language}.jsonl")

        assert os.path.exists(self.input_dev_file), "Test file not found: {}".format(
            self.input_dev_file
        )
        self.output_dev_file = os.path.join(output_dir, "dev.jsonl" if language is None else f"dev_{language}.jsonl")

        if self.input_test_backtranslation_file is not None and not os.path.exists(self.input_test_backtranslation_file):
            print(f"Test Backtranslation file not found: {self.input_test_backtranslation_file}")
            self.output_test_backtranslation_file = None
        else:
            self.output_test_backtranslation_file = os.path.join(output_dir, f"test_en_{language}_bt.json")


        self.glove_mapping = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
        }

    def run(self):
        print("Converting dataset to jsonl")
        os.makedirs(self.output_dir, exist_ok=True)
        self._run_normally()

    def _run_normally(self):
        # Convert the dev and test set
        self._convert_tacred_format_file(self.input_test_file, self.output_test_file)
        self._convert_tacred_format_file(self.input_dev_file, self.output_dev_file)
        self._convert_tacred_format_file(self.input_train_file, self.output_train_file)
        if self.output_test_backtranslation_file:
            self._convert_tacred_format_file(self.input_test_backtranslation_file, self.output_test_backtranslation_file)

    def _convert_tacred_format_file(self, input_file, output_file):
        with open(output_file, "w", encoding="utf-8") as output_file:
            for example in self._read_tacred_file(input_file):
                output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _read_tacred_file(self, input_file):
        with open(input_file, "r", encoding="utf-8") as input_file:
            input_examples = json.loads(input_file.readline())
            for input_example in input_examples:
                tokens = input_example["token"]
                subj_offsets = (input_example["subj_start"], input_example["subj_end"] + 1)
                obj_offsets = (input_example["obj_start"], input_example["obj_end"] + 1)

                tokens = self.normalize_glove_tokens(tokens)

                output_example = {
                    "id": input_example["id"],
                    "tokens": tokens,
                    "label": input_example["relation"],
                    "entities": (subj_offsets, obj_offsets),
                    "grammar": ("SUBJ", "OBJ"),
                    "type": (input_example["subj_type"], input_example["obj_type"]),
                }

                yield output_example

    def normalize_glove_tokens(self, tokens):
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
        "--dataset_dir", type=str, help="The data/json directory of the TACRED dataset"
    )
    parser.add_argument("--output_dir", type=str, help="An output directory of jsonl files")
    parser.add_argument("--language", type=str, help="An optional language 2-letter-code identifier", required=False)

    args = parser.parse_args()
    print(args)
    main(args)
