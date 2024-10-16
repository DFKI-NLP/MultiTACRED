# MultiTACRED: A Multilingual Version of the TAC Relation Extraction Dataset

This repository contains the code of our paper:
[MultiTACRED: A Multilingual Version of the TAC Relation Extraction Dataset.](https://arxiv.org/abs/2305.04582)
Leonhard Hennig, Philippe Thomas, Sebastian Möller

We machine-translate the TAC relation extraction dataset [1] to 12 typologically diverse languages from
different language families, analyze translation and annotation projection quality, and evaluate
fine-tuned mono- and multilingual PLMs in common transfer learning scenarios.

- **HF dataset reader**: https://huggingface.co/datasets/DFKI-SLT/multitacred
- **Papers With Code**: https://paperswithcode.com/dataset/multitacred
- **LDC**: https://catalog.ldc.upenn.edu/LDC2024T09

# Access

To respect the copyright of the underlying TACRED and KBP corpora, MultiTACRED is released via the Linguistic Data Consortium (LDC). Therefore, you can download MultiTACRED from the [LDC MultiTACRED webpage](https://catalog.ldc.upenn.edu/LDC2024T09). If you are an LDC member, the access will be free; otherwise, an access fee of $25 is needed.


# Installation

---

## 🔭&nbsp; Overview


## ✅&nbsp; Requirements

MultiTACRED is tested with:

- Python >= 3.8
- Torch >= 1.10.2; <= 1.12.1
- AllenNLP >= 2.8.0; <= 2.10.1
- Transformers >= 4.12.5; <= 4.20.1

## 🚀&nbsp; Installation

### From source
```bash
git clone https://github.com/DFKI-NLP/MultiTACRED
cd MultiTACRED
pip install .
```

## 🔧&nbsp; Usage

### Preparing the TACRED dataset
In order to run the translation scripts, we need to convert the files from the LDC-provided JSON format
to a simpler JSONL format:
```bash
python src/translate/convert_to_jsonl.py --dataset_dir [/path/to/tacred/data/json] --output_dir ./data/en
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
sample file `data/en/train_sample.jsonl` to German:

```bash
python src/translate/translate_deepl.py --api_key [API_KEY] --api_address "https://api.deepl.com/v2/translate"
-i ./data/en/train_sample.jsonl -o ./data/de/train_de_deepl.jsonl -T spacy_de -s EN -t DE
--log_file translate.log
```
Your output should be similar to `data/de/train_de_sample.jsonl`. Note that the output file contains
the original English tokens (field `tokens`) as well as the translated tokens (field `tokens_translated`) and the raw translation (field `translation_raw`).

For testing purposes, you can use the character-limited free API endpoint
[https://api-free.deepl.com/v2/translate](https://api-free.deepl.com/v2/translate) (but you still need an API key).

Call `python src/translate/translate_deepl.py --help` for usage and argument information.

For a list of available languages use the flag `--show_languages`.

#### Google

The script [`translate_google.py`](translate_google.py) translates the dataset into the target
language. You need a valid API key. Follow this [setup guide](https://cloud.google.com/translate/docs/setup) and
place the private key in a secure location.

The following example shows how to translate the sample file `data/en/train_sample.jsonl` to German:

```bash
python src/translate/translate_google.py --private_key [path/to/PRIVATE_KEY]
-i ./data/en/train_sample.jsonl -o ./data/de/train_de_google.jsonl -T spacy_de -s EN -t DE
--log_file translate.log
```

Call `python src/translate/translate_google.py --help` for usage and argument information.

For a list of available langugages use the flag `--show_languages`.

#### Using an .env file
For convenience, it is best to create a .env file of following content:

```
INPUT_FILE='/path/to/dataset.jsonl'
OUTPUT_FILE='dataset_translated.jsonl'
# For DeepL
API_KEY="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:xx"
API_ADDRESS="https://api.deepl.com/v2/translate"
# or use: API_ADDRESS="https://api-free.deepl.com/v2/translate" (limited to 500K chars/month)

# For Google
PRIVATE_KEY='/path/to/private_key.json'
```

#### Tokenizer

The argument `-T` or `--tokenizer` gives you a choice between which tokenizer
you want to use to tokenize the raw translation text.

- split (default): python whitespace tokenization
- spacy [website](https://spacy.io/usage), the used language model will be downloaded automatically (currently, the statistical models, not the neural ones).
- trankit [website](https://github.com/nlp-uoregon/trankit), the used neural model will be downloaded automatically. Requires a GPU to run at reasonable speed!

You can add your own tokenizer by just adding the tokenizer name to
`tokenizer_choices` and its initialization in the `init_tokenizer()` function of the
[utils.py](src/translate/utils.py) script.

#### Logging

The scripts implement logging from the python standard library.
- Set `-v` to display logging messages to the console
- Set `--log_level [debug,info,warning,error,critical]` to determine which kind things should be logged.
- Set `--log_file FILE` to log to a file.
- Alternatively give a logger object as argument to the translate function.

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

### Scripts to wrap Translation, Backtranslation and Conversion to JSON
`scripts/translate_deepl.sh` and `scripts/translate_google.sh` wrap translation,
backtranslation, and conversion to JSON for a single language. You do still need
to do the one-time-only step of preparing the JSONL version of the original TACRED!


## Relation Extraction Experiments
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
the TACRED json files. We provide a slightly modified version of the original `apply_tacred_patch.py` script 
to account for non-ascii characters (json dump with ensure_ascii=False) and the reduced amount of instances in 
dev / test due to translation errors (remove an id check assertion).

```bash
git clone https://github.com/DFKI-NLP/tacrev
```

Then, for any language, run:
```bash
# Dev split
python ./scripts/apply_tacrev_patch.py \
  --dataset-file ./data/[lang]/dev_[lang].json \
  --patch-file [path/to/tacrev/patch/dev_patch.json] \
  --output-file ./data/[lang]/dev_[lang].json
  
# Test split
python ./scripts/apply_tacrev_patch.py \
  --dataset-file ./data/[lang]/test_[lang].json \
  --patch-file [path/to/tacrev/patch/test_patch.json] \
  --output-file ./data/[lang]/test_[lang].json

# Backtranslated Test split
python ./scripts/apply_tacrev_patch.py \
  --dataset-file ./data/[lang]/test_en_[lang]_bt.json \
  --patch-file [path/to/tacrev/patch/test_patch.json] \
  --output-file ./data/[lang]/test_en_[lang]_bt.json
```

### Train and evaluate a single scenario
```bash
python src/evaluate/evaluate.py "scenario=inlang_de"
```

### Train and evaluate a single scenario, with different random seeds
```bash
python src/evaluate/evaluate.py "run_id=range(1,6)" "scenario=inlang_de" -m
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
@inproceedings{hennig-etal-2023-multitacred,
    title = "MultiTACRED: A Multilingual Version of the TAC Relation Extraction Dataset",
    author = "Hennig, Leonhard and Thomas, Philippe and Möller, Sebastian",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Online and Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    abstract = "Relation extraction (RE) is a fundamental task in information extraction, whose extension to multilingual settings has been hindered by the lack of supervised resources comparable in size to large English datasets such as TACRED (Zhang et al., 2017). To address this gap, we introduce the MultiTACRED dataset, covering 12 typologically diverse languages from 9 language families, which is created by machine-translating TACRED instances and automatically projecting their entity annotations. We analyze translation and annotation projection quality, identify error categories, and experimentally evaluate fine-tuned pretrained mono- and multilingual language models in common transfer learning scenarios. Our analyses show that machine translation is a viable strategy to transfer RE instances, with native speakers judging more than 84\% of the translated instances to be linguistically and semantically acceptable. We find monolingual RE model performance to be comparable to the English original for many of the target languages, and that multilingual models trained on a combination of English and target language data can outperform their monolingual counterparts. However, we also observe a variety of translation and annotation projection errors, both due to the MT systems and linguistic features of the target languages, such as pronoun-dropping, compounding and inflection, that degrade dataset quality and RE model performance.",
}
```

## 📘&nbsp; Licence
MultiTACRED is released under the terms of the [MIT License](LICENCE).

## References
1. [Position-aware Attention and Supervised Data Improve Slot Filling](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf). Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D. Manning.
