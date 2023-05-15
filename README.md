# MultiTACRED: A Multilingual Version of the TAC Relation Extraction Dataset

This repository contains the code of our paper:
[MultiTACRED: A Multilingual Version of the TAC Relation Extraction Dataset.](https://arxiv.org/abs/2305.04582)
Leonhard Hennig, Philippe Thomas, Sebastian Möller

We machine-translate the TAC relation extraction dataset [1] to 12 typologically diverse languages from
different language families, analyze translation and annotation projection quality, and evaluate
fine-tuned mono- and multilingual PLMs in common transfer learning scenarios.

# Installation

---

## 🔭&nbsp; Overview


## ✅&nbsp; Requirements

MultiTACRED is tested with:

- Python 3.8
- Torch 1.10.2
- AllenNLP 2.8.0
- Transformers 4.12.5

## 🚀&nbsp; Installation

### From source
```bash
git clone https://github.com/DFKI-NLP/MultiTACRED.git
cd MultiTACRED
pip install .
```

## 🔧&nbsp; Usage

### Preparing the TACRED dataset
In order to run the translation scripts, we need to convert the files from the LDC-provided JSON format
to a simpler JSONL format:
```bash
cp *.json [/path/to/tacred/data/json] ./data/en/
python src/translate/convert_to_jsonl.py --dataset_dir ./data/en --output_dir ./data/en
```

### Translation
Translation uses the [DeepL](https://www.deepl.com/pro-api?cta=header-pro/)
or [Google](https://cloud.google.com/translate/?hl=en) APIs.

`translate_deepl.py` and `translate_google.py` translate a `.jsonl` dataset into a different language.
The dataset is expected to be in following format (i.e. the JSONL format created above):
```
{"id": original_id, "tokens": [original_tokens], "label": [original_label], "entities": [original_entities], "grammar": [original_grammar], "type": [original_type]}
```

The translated result in `[output_file.jsonl]` appears in the following form:
```
{"id": original_id, "tokens": [original_tokens], "label": [original_label], "entities": [original_entities], "grammar": [original_grammar], "type": [original_type], "language": [original_language], "tokens_translated": [translated_tokens], "entities_translated": [translated_entities], "language_translated": translated_language, "text_raw": [original_text], "translation_raw": [raw_translation_text]}
```

The scripts additionally create a file `[output_file.jsonl].manual`. This file contains all examples in which the
script fails to extract the entities looking at the number and ordering of the entities. If debugging level is set to 'warning',
the logger creates warnings for any such example.

The scripts skip translation for all examples that are in `[output_file.jsonl]`
and `[output_file.jsonl].manual` to avoid costly unnecessary translation. Set `--overwrite` to
re-translate those examples.


#### DeepL
The script [`translate_deepl.py`](translate_deepl.py) translates the dataset into the
target language. You need a valid API key. The following example shows how to translate the
first 1000 characters of `train.jsonl` to German:

```bash
python src/translate/translate_deepl.py --api_key [API_KEY] --api_address https://api.deepl.com/v2/translate
-i ./data/en/train.jsonl -o ./data/de/tacred_train_de.jsonl -T spacy_de -s EN -t DE
--log_file translate.log --max_characters 1000
```
For testing purposes, you can also use the character-limited free API endpoint
[https://api-free.deepl.com/v2/translate](https://api-free.deepl.com/v2/translate) (but you still need an API key).

Call `python src/translate/translate_deepl.py --help` for usage and argument information.

For a list of available languages use the flag `--show_languages`.

#### Google

The script [`translate_google.py`](translate_google.py) translates the dataset into the target
language. You need a valid API key. The following example shows how to translate the
first 1000 characters of `train.jsonl` to German:

```bash
python src/translate/translate_google.py --private_key [path/to/PRIVATE_KEY]
-i ./data/en/train.jsonl -o ./data/de/tacred_train_de.jsonl -T spacy_de -s EN -t DE
--log_file translate.log --max_characters 1000
```

Follow this [setup guide](https://cloud.google.com/translate/docs/setup) and
place the private key in a secure location.

Call `python src/translate/translate_google.py --help` for usage and argument information.

For a list of available langugages use the flag `--show_languages`.

#### Using an .env file
For convenience, it is best to create a .env file of following content:

```
INPUT_FILE='/path/to/dataset.jsonl'
OUTPUT_FILE='dataset_translated.jsonl'
# For DeepL
API_KEY='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:xx'
API_ADDRESS='https://api.deepl.com/v2/translate'
# or use: API_ADDRESS='https://api-free.deepl.com/v2/translate' (limited to 500K chars/month)

# For Google
PRIVATE_KEY='/path/to/private_key.json'
```

### Backtranslation
The script ['backtranslate.py'](backtranslate.py) translates a translated dataset back to its
original language. The function accepts output.jsonl files in the format of the
`translate_deepl.py` and `translate_google.py` file. In this case it is necessary
to specify an input, output file and a service:

```
python src/translate/backtranslate.py [translated.jsonl] [backtranslated.jsonl] [google|deepl]
```
Additionally, the script accepts the same arguments (and .env file) as
`translate_deepl.py` and `translate_google.py` because it uses one of both.

The script is a wrapper for a sequence of calls of scripts. It calls following scripts in this order:
1. `prepare_backtranslation.py` creates
a temporary file (default=`.temp_[input_file]`) with the right format to call the translation script on (switching field names)
2. `translate_deepl.py` or `translate_google.py` are called on the temporary file,
the outputs are found in two temporary files
(default=`.temp_[output_file]` and `.temp_[output_file].manual`).
**Be careful about deleting them**, as at it is here where all the raw
backtranslation results accumulate.
3. `postpare_backtranslation.py` all entries in the temporary file back into an
acceptable format

### Converting JSONL to TACRED JSON
If you require the translations in the orginal TACRED JSON format, e.g. when using the
[Huggingface Tacred DatasetReader](https://huggingface.co/datasets/DFKI-SLT/tacred),
you can call the script:
```bash
python src/translate/convert_to_json.py --dataset_dir [/path/to/translated/jsonl] --output_dir [json-output-dir] --language [lang_code]
```

### Scripts to wrap all this
`scripts/translate_deepl.sh` and `scripts/translate_google.sh` wrap translation,
backtranslation, and conversion to JSON for a single language. You do still need
to do the one-time-only step of preparing the JSONL version of the original TACRED!

### Tokenizer

The argument `-T` or `--tokenizer` gives you a choice between which tokenizer
you want to use to tokenize the raw translation text.

- split (default): python whitespace tokenization
- spacy [website](https://spacy.io/usage), the used language model will be downloaded automatically (currently, the statistical models, not the neural ones).
- trankit [website](https://github.com/nlp-uoregon/trankit), the used neural model will be downloaded automatically. Requires a GPU to run at reasonable speed!

You can add your own tokenizer by just adding the tokenizer name to
`tokenizer_choices` and its initialization in the `init_tokenizer()` function of the
[utils.py](src/translate/utils.py) script.

### Logging

The scripts implement logging from the python standard library.
- Set `-v` to display logging messages to the console
- Set `--log_level [debug,info,warning,error,critical]` to determine which kind things should be logged.
- Set `--log_file FILE` to log to a file.
- Alternatively give a logger object as argument to the translate function.


## Experiments
All experiments are configured with [Hydra](https://hydra.cc/). Experiment configs are
stored in `config/`. 

### Preparing the data
You need to obtain the MultiTACRED dataset from [this URL](https://ldc.upenn.edu/TODO), 
and unzip it into the `./data` folder.
You also need to download the original, [English TACRED dataset](https://catalog.ldc.upenn.edu/LDC2018T24),
and place the content of its `data/json` folder in `./data/en` 

The file structure should look like this:
```bash
data
  |-- ar/
       |--- train_ar.json
       |--- dev_ar.json
       |--- test_ar.json
       |--- test_en_ar_bt.json
  |-- de
       |--- ...
  |-- en
       |--- train.json
       |--- dev.json
       |--- test.json
  |...
```

To reproduce our results, you should apply the [TACRED Revisited](https://github.com/DFKI-NLP/tacrev) patch to
the English TACRED json files. Note that the translated data is already patched.
```bash
git clone https://github.com/DFKI-NLP/tacrev

python tacrev/scripts/apply_tacred_patch.py \
  --dataset-file TACRED/data/json/dev.json \
  --patch-file tacrev/patch/dev_patch.json \
  --output-file MultiTACRED/data/dev.json

python tacrev/scripts/apply_tacred_patch.py \
  --dataset-file TACRED/data/json/test.json \
  --patch-file tacrev/patch/test_patch.json \
  --output-file MultiTACRED/data/test.json
```

### Train and evaluate a single scenario
```bash
python src/evaluate/evaluate.py "scenario=inlang_de"
```

### Train and evaluate a single scenario, with different random seeds
```bash
python src/evaluate/evaluate.py "run_id=range(1,6)" "scenario=inlang_de"
```

### Train and evaluate all in-language and cross-language scenarios
```bash
python src/evaluate/evaluate.py "run_id=range(1,6)" "scenario=glob([inlang*,crosslang*])" -m
```

### Train and evaluate the multi-language scenarios
```bash
python src/evaluate/evaluate.py "run_id=range(1,6)" "scenario=glob(multilang*)" "scenario.additional_train_dev_files_sample_size=range(0.1,1.1,0.1)" "+experiment=multilang" -m
```

### Find the best learning rate
```bash
- python src/evaluate/evaluate.py "learning_rate=3e-6,7e-6,1e-5,3-e5,5e-5" "scenario=glob([inlang*,crosslang_en])" "+experiment=learning_rate" -m
```

### Evaluate the backtranslations
```bash
python src/evaluate/evaluate.py "run_id=range(1,6)" "scenario=glob(backtranslation*)" -m
```

## 📚&nbsp; Citation
Please consider citing the following paper when using MultiTACRED:
```
@misc{hennig-etal-2023-multitacred,
  author = {Leonhard Hennig, Philippe Thomas},
  title = {MultiTACRED: A Multilingual Version of the TAC Relation Extraction Dataset.},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DFKI-NLP/MultiTACRED}}
}
```

## 📘&nbsp; Licence
TranslateRE is released under the terms of the [MIT License](LICENCE).

## References
1. [Position-aware Attention and Supervised Data Improve Slot Filling](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf). Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D. Manning.
