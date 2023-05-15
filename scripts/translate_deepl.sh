#!/bin/bash
TOKENIZER="spacy_de"
SOURCE_LANG="EN"
TARGET_LANG="DE"
TARGET_LANG_LC="de"
API_KEY=[api-key]
API_ADDRESS="https://api.deepl.com/v2/translate"
MAX_CHARACTERS=-1
TACRED_PATH="./data/en"
OUTPUT_DIR="./data"
GPU=0
export PYTHONPATH="$PYTHONPATH:."

echo 'Config'
echo $SOURCE_LANG '->' $TARGET_LANG
echo 'Tokenizer ' $TOKENIZER
echo 'Max characters ' $MAX_CHARACTERS

mkdir -p ${OUTPUT_DIR}/${TARGET_LANG_LC}
for SPLIT in 'train' 'dev' 'test'
do
  echo 'Translating split: ' $SPLIT
  CUDA_VISIBLE_DEVICES=${GPU} python src/translate/translate_deepl.py -v --api_key ${API_KEY} --api_address ${API_ADDRESS} \
  -i ${TACRED_PATH}/${SPLIT}.jsonl -o ${OUTPUT_DIR}/${TARGET_LANG_LC}/${SPLIT}_${TARGET_LANG_LC}.jsonl \
  -T ${TOKENIZER} -s ${SOURCE_LANG} -t ${TARGET_LANG} --max_characters ${MAX_CHARACTERS} \
  --log_file ${OUTPUT_DIR}/${TARGET_LANG_LC}/${SOURCE_LANG}-${TARGET_LANG}.log
done
echo 'Backtranslating Test split'
CUDA_VISIBLE_DEVICES=${GPU} python src/translate/backtranslate.py $OUTPUT_DIR/${TARGET_LANG_LC}/test_${TARGET_LANG_LC}.jsonl \
  ${OUTPUT_DIR}/${TARGET_LANG_LC}/test_en_${TARGET_LANG_LC}_bt.jsonl deepl \
  -v --api_key ${API_KEY} --api_address ${API_ADDRESS} \
  -T spacy_en --max_characters ${MAX_CHARACTERS} \
  --log_file ${OUTPUT_DIR}/${TARGET_LANG_LC}/${SOURCE_LANG}-${TARGET_LANG}_backtranslation.log

echo 'Converting to TACRED JSON'
python src/translate/convert_to_json.py --dataset_dir ${OUTPUT_DIR}/${TARGET_LANG_LC} \
  --output_dir ${OUTPUT_DIR}/${TARGET_LANG_LC} --language ${TARGET_LANG_LC}
echo 'Done'
