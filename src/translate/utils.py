#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 03.11.21
@author: gabriel.kressin@dfki.de
@description: functions for dataset translation
"""
import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

TOKEN_HEAD_S = "<H>"
TOKEN_HEAD_E = "</H>"
TOKEN_TAIL_S = "<T>"
TOKEN_TAIL_E = "</T>"
LOG_CHOICES = ["debug", "info", "warning", "error", "critical"]
MAX_CONSECUTIVE_ERRORS = 10
SPACY_MODELS = {
    "spacy_de": "de_core_news_sm",
    "spacy_en": "en_core_web_sm",
    "spacy_pl": "pl_core_news_sm",
    "spacy_zh": "zh_core_web_sm",
    "spacy_es": "es_core_news_sm",
    "spacy_ja": "ja_core_news_sm",
    "spacy_el": "el_core_news_sm",  # greek
    "spacy_pt": "pt_core_news_sm",  # portuguese portuguese, not brazilian?
    "spacy_fr": "fr_core_news_sm",
    "spacy_lt": "lt_core_news_sm",  # lithuanian
    "spacy_ru": "ru_core_news_sm",
    "spacy_nl": "nl_core_news_sm",
    "spacy_fi": "fi_core_news_sm",
}

TRANKIT_MODELS = {
    "trankit_hu": "hungarian",
    "trankit_ar": "arabic",
    "trankit_tr": "turkish",
    "trankit_hi": "hindi",
    "trankit_eu": "basque",
}


def init_logger(
    name,  # Name of initializing script
    verbose,  # Bool, whether logger prints to console or not
    log_level_string,  # one of LOG_CHOICES
    log_file,  # path to log_file, works with None
):
    logger_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    idx_choice = LOG_CHOICES.index(log_level_string)
    log_level = logger_levels[idx_choice]

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def return_without_char_at_index(string, index):
    """Returns string without char at index."""
    return string[:index] + string[index + 1 :]


def join_tags(tokenized_text):
    """Loops through tokens and joins <H>,.. tags preserving segmentation elsewhere."""

    tags = (TOKEN_HEAD_S, TOKEN_HEAD_E, TOKEN_TAIL_S, TOKEN_TAIL_E)
    candidate = []
    idx_candidate = 0
    first_candidate_position = None
    candidate_segments = []
    result = []
    full_match = False
    for segment in tokenized_text:
        for idx_char, char in enumerate(segment):
            matched = False
            # Try to match any possible tag
            for tag in tags:
                if idx_candidate < len(tag) and tag[idx_candidate] == char:

                    # only add char to candidates once.
                    if not matched:
                        candidate.append(char)
                        matched = True

                    # remember first canditate position, so it can be deleted afterwards.
                    if first_candidate_position is None:
                        first_candidate_position = idx_char

                    # check whether a tag has been found
                    if tag == "".join(candidate):
                        full_match = True

                        # case: tag is within a unsplitted sequence
                        if not len(candidate_segments):
                            candidate_segments.append(segment[:first_candidate_position])
                        # Delete matched chars from first previous segment, discard others
                        processed_first_candidate_segment = candidate_segments[0][
                            :first_candidate_position
                        ]
                        if len(processed_first_candidate_segment):
                            result.append(processed_first_candidate_segment)
                        candidate_segments = []
                        result.append(tag)

                        # reset, keep going from beginning with adapted restsegment
                        rest_segment = segment[idx_char + 1 :]
                        matched = False

            # If no consecutive match, reset
            if not matched:
                candidate = []
                idx_candidate = 0
                first_candidate_position = None
                result.extend(candidate_segments)
                candidate_segments = []
            else:
                idx_candidate += 1

        # prevent appending entire segment for full_match
        if full_match:
            full_match = False
            segment = rest_segment
        # If segment is "open", save it
        if matched and len(segment):
            candidate_segments.append(segment)
        # Else, append it to result
        else:
            if len(segment):
                result.append(segment)

    return result


def exclude_double_entity_tags(tokenized_text, regex):
    """Removes entity from text if it's text matches with a regex.

    Internally the text between <H> and </H> or <T> and </T> is joined and
    matched to the regex, only it fully matches the regex, the entity is removed.

    Note this is pretty inefficient, as it is the same pass as the post_process code later
    but it works and is easier to understand than hacking some conditional in the code later
    on.
    """

    tokens_translated = []
    indices_start_end_tokens = [[], []]
    token_offset = 0
    # 1. seperate tags from text
    for (idx, token) in enumerate(tokenized_text):
        append_after = []
        # add start tokens directly, to tokens, update offset.
        if TOKEN_HEAD_S in token:
            indices_start_end_tokens[0].append(idx - token_offset)
            token = token.replace(TOKEN_HEAD_S, "")
            tokens_translated.append(TOKEN_HEAD_S)
            token_offset -= 1
        if TOKEN_HEAD_E in token:
            token = token.replace(TOKEN_HEAD_E, "")
            append_after.append(TOKEN_HEAD_E)
        if TOKEN_TAIL_S in token:
            indices_start_end_tokens[1].append(idx - token_offset)
            token = token.replace(TOKEN_TAIL_S, "")
            tokens_translated.append(TOKEN_TAIL_S)
            token_offset -= 1
        if TOKEN_TAIL_E in token:
            token = token.replace(TOKEN_TAIL_E, "")
            append_after.append(TOKEN_TAIL_E)

        # remove whitespace tokens
        if len(token.strip()) > 0:
            tokens_translated.append(token)
        else:
            token_offset += 1

        # add end tokens after adding the entity-token. Update offset too.
        for end_tag in append_after:
            tokens_translated.append(end_tag)
            if TOKEN_HEAD_E == end_tag:
                indices_start_end_tokens[0].append(idx + 1 - token_offset)
                token_offset -= 1
            else:
                indices_start_end_tokens[1].append(idx + 1 - token_offset)
                token_offset -= 1

    # determine unwanted tags indices
    tags_to_remove = []
    for entity_indices in indices_start_end_tokens:
        # Only handle even tag cases.
        if len(entity_indices) > 2 and len(indices_start_end_tokens[0]) % 2 == 0:
            # Make sure one tag pair always stays, by default it will be the latter one
            indices_left = len(entity_indices)
            for idx in range(0, len(indices_start_end_tokens), 2):
                # assume that order is preserved <T>...</T> <T>...</T>
                start = entity_indices[idx]
                end = entity_indices[idx + 1]
                entity_text = " ".join(tokens_translated[start + 1 : end])
                if re.fullmatch(regex, entity_text) and indices_left > 2:
                    tags_to_remove.extend((start, end))
                    indices_left -= 2

    # remove unwanted tags
    return [
        token
        for token_idx, token in enumerate(tokens_translated)
        if token_idx not in tags_to_remove
    ]


# Add your tokenizer to this list
TOKENIZER_CHOICES = ["split"] + list(SPACY_MODELS.keys()) + list(TRANKIT_MODELS.keys())


def init_tokenizer(logger, tokenizer_name):
    """
    Initializes tokenizer based on name

    Parameters
    ----------
    logger : logging.logger
        logging object from calling script
    tokenizer_name : string
        specifiying which logger to load

    Returns
    -------
    function (string) -> list
        function that splits text given as argument
        into tokens. Returns those tokens as a list of strings.
    """

    if tokenizer_name == "split":
        return lambda text: text.split()
    elif tokenizer_name in SPACY_MODELS.keys():  # == 'spacy_de':
        import re

        import spacy

        try:
            nlp = spacy.load(SPACY_MODELS[tokenizer_name])
        except OSError:
            from spacy.cli.download import download as spacy_download

            spacy_download(SPACY_MODELS[tokenizer_name])
            nlp = spacy.load(SPACY_MODELS[tokenizer_name])

        # Handle chinese/japanese tokenizer seperately (see README.md)
        if tokenizer_name == "spacy_zh" or tokenizer_name == "spacy_ja":
            return lambda text: join_tags([token.text for token in nlp.tokenizer(text)])

        # Add html tags and escaped tokens as exception
        token_match_re = re.compile(r"(</?.>)|(&[l|g]t;)")
        # Add rule to split on hyphens
        # infixes = nlp.Defaults.infixes + ["-",]
        # infix_re = spacy.util.compile_infix_regex(infixes)
        # Prevent '<' from being split
        prefixes = nlp.Defaults.prefixes
        if ("<") in prefixes:
            prefixes.remove("<")  # suffix/prefix override token_match sometimes
        prefix_re = spacy.util.compile_prefix_regex(prefixes)
        suffixes = nlp.Defaults.suffixes

        # Prevent '>' and '/' from being split
        if ">" in suffixes:
            suffixes.remove(">")
        if "/" in suffixes:
            suffixes.remove("/")  # Some links end on / (e.g. id=e779865fb9e00b8b534b)
        suffix_re = spacy.util.compile_suffix_regex(suffixes)

        nlp.tokenizer.token_match = token_match_re.match
        nlp.tokenizer.prefix_search = prefix_re.search
        nlp.tokenizer.suffix_search = suffix_re.search
        # nlp.tokenizer.infix_finditer = infix_re.finditer

        if tokenizer_name == "spacy_es":
            # Sometimes el or la get marked as entities. Exclude such entities.
            regex = "(?i)el|la"
            return lambda text: exclude_double_entity_tags(
                [token.text for token in nlp.tokenizer(text)], regex
            )

        return lambda text: [token.text for token in nlp.tokenizer(text)]

    elif tokenizer_name in TRANKIT_MODELS.keys():
        from trankit import Pipeline

        pipeline = Pipeline(TRANKIT_MODELS[tokenizer_name], gpu=True)
        return lambda text: join_tags(
            [t["text"] for s in pipeline.tokenize(text)["sentences"] for t in s["tokens"]]
        )

    elif tokenizer_name == "other_tokenizer":
        # Add your tokenizer here, and in TOKENIZER_CHOICES above the function
        raise Exception("not implemented")
    else:
        raise Exception(f"Please choose one of {TOKENIZER_CHOICES}")


def escape_token(token):
    # return html.escape(token).replace("`", "&lsquo;").replace("'", "&rsquo;")
    # why not escape & to &amp; ? or any other html escapes?
    return token.replace("<", "&lt;").replace(">", "&gt;")


# def unescape_token(token):
#    #return html.unescape(token.replace("&lsquo;", "`").replace("&rsquo;", "'"))
#    # Example 61b3a65fb99558e066d7 - Deepl returns '&amp;', which needs to be unescaped
#    return token.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')


def pre_process(example):
    """
    Returns a string from an example in json/dict format
    """
    tokens = [escape_token(t) for t in example["tokens"]]
    head = example["entities"][0]
    tail = example["entities"][1]

    tokens[head[0]] = TOKEN_HEAD_S + tokens[head[0]]
    tokens[head[1] - 1] = tokens[head[1] - 1] + TOKEN_HEAD_E
    tokens[tail[0]] = TOKEN_TAIL_S + tokens[tail[0]]
    tokens[tail[1] - 1] = tokens[tail[1] - 1] + TOKEN_TAIL_E

    text = " ".join(tokens)
    return text


def post_process(text, tokenizer):
    """
    Returns sentence in unescaped tokens and positions of head and tail.
    """
    # the fixes are pretty hacky and should be tested
    text = html.unescape(text)  # fixes HTML entities
    # Google sometimes splits ending tokens up: </H> -> </ H>
    text = (
        text.replace("</ H>", "</H>")
        .replace("</ T>", "</T>")
        .replace("< /H>", "</H>")
        .replace("< /T>", "</T>")
    )
    # Spacy tokenizer tokenizes 'sometext</H>,' as a single token, fix by inserting whitespace, but remove a
    # text-ending whitespace
    # also replace some errors after introducing commas in German translation, e.g. '<H>, Anwar Chowdry</H>'
    text = (
        text.replace("<H>", " <H>")
        .replace("</H>", "</H> ")
        .replace("<T>", " <T>")
        .replace("</T>", "</T> ")
        .replace("<H>, ", ", <H>")
        .replace("<T>, ", ", <T>")
        .strip()
    )

    tokens = tokenizer(text)

    tokens_translated = []
    entities_translated = [[], []]
    ws_tokens = 0
    for (idx, token) in enumerate(tokens):
        if TOKEN_HEAD_S in token:
            entities_translated[0].append(idx - ws_tokens)
            token = token.replace(TOKEN_HEAD_S, "")
        if TOKEN_HEAD_E in token:
            token = token.replace(TOKEN_HEAD_E, "")
            if len(token.strip()) > 0:
                entities_translated[0].append(idx + 1 - ws_tokens)
            else:
                # if </H> was the only token, do not want to add extra index
                entities_translated[0].append(idx - ws_tokens)
        if TOKEN_TAIL_S in token:
            entities_translated[1].append(idx - ws_tokens)
            token = token.replace(TOKEN_TAIL_S, "")
        if TOKEN_TAIL_E in token:
            token = token.replace(TOKEN_TAIL_E, "")
            if len(token.strip()) > 0:
                entities_translated[1].append(idx + 1 - ws_tokens)
            else:
                # if </T> was the only token, do not want to add extra index
                entities_translated[1].append(idx - ws_tokens)

        # remove whitespace tokens
        if len(token.strip()) > 0:
            tokens_translated.append(token)
        else:
            ws_tokens += 1

    if (
        len(entities_translated) == 2
        and len(entities_translated[0]) > 1
        and len(entities_translated[1]) > 1
    ):
        if entities_translated[0][1] > len(tokens_translated):
            entities_translated[0][1] = len(tokens_translated)
        if entities_translated[1][1] > len(tokens_translated):
            entities_translated[1][1] = len(tokens_translated)
        assert 0 <= entities_translated[0][0] < entities_translated[0][1]
        assert 0 <= entities_translated[0][1] <= len(tokens_translated)
        assert 0 <= entities_translated[1][0] < entities_translated[1][1]
        assert 0 <= entities_translated[1][1] <= len(tokens_translated)
    return tokens_translated, entities_translated


def entities_check(entities):
    """superficially check if entity spans match"""
    return (
        len(entities[0]) == 2
        and len(entities[1]) == 2
        and entities[0][0] < entities[0][1]
        and entities[1][0] < entities[1][1]
        and (entities[0][1] <= entities[1][0] or entities[1][1] <= entities[0][0])
    )


def create_file(logger, filename, overwrite_translated):
    try:
        if not os.path.exists(Path(filename).parent):
            os.makedirs(Path(filename).parent)
        with open(filename, "x", encoding="utf-8"):
            logger.info(f"Created file {filename}")
    except FileExistsError:
        if overwrite_translated:
            logger.warning(f"Overwriting existing file: {filename}")
        else:
            logger.info(f"Appending to existing file: {filename}")


def translate_and_write(
    logger,
    texts,  # list with strings to be translated
    examples,  # matching list of dictionaries with example info
    tokenizer,
    translation_function,
    source_language,
    target_language,
    f_out,  # file_object output_file
    fm_out,  # file_object output_file.manual
):
    """
    Returns (success, error_object, counts_translated, counts_manual)
    """
    success, response, key, err = translation_function(texts, source_language, target_language)
    if success:
        count_manual = 0
        for (idx, translation) in enumerate(response):
            text = translation[key]
            tokens_translated, entities_translated = post_process(text, tokenizer)
            examples[idx]["language"] = source_language
            examples[idx]["tokens_translated"] = tokens_translated
            examples[idx]["entities_translated"] = entities_translated
            examples[idx]["language_translated"] = target_language
            examples[idx]["text_raw"] = texts[idx]
            examples[idx]["translation_raw"] = text

            if entities_check(entities_translated):
                f_out.write(json.dumps(examples[idx], ensure_ascii=False) + "\n")
            else:
                count_manual += 1
                # Append plain text for better manual fixing
                fm_out.write(json.dumps(examples[idx], ensure_ascii=False) + "\n")
                logger.warning(
                    "Error processing example with"
                    f' id={examples[idx]["id"]}, written to'
                    f" .manual"
                )
        return True, None, len(texts) - count_manual, count_manual
    else:
        return False, err, 0, 0


def translate_dataset(
    logger,
    translation_function,  # function that translates : API key for deepl API
    input_file,  # string: path/to/input/file
    output_file,  # string: path/to/output/file
    tokenizer,
    source_language="EN",
    target_language="DE",
    overwrite_translated=False,  # whether old translations in output_file shold be overwritten
    max_characters=-1,  # Maximum characters translated (to save the quota) -1 = disable
    pre_process_func=pre_process,
):
    """
    Takes the dataset and translates it from the source to the target
    language.
    Creates two files:
        output_file(.jsonl) :
            File containing translated dataset (as .jsonl)
        output_file(.jsonl).manual :
            File containing examples of the dataset in which the post processing
            for the given translation failed.
    """

    # Handle output_file
    create_file(logger, output_file, overwrite_translated)

    # Handle manual output_file_manual:
    # file with all wrongly postprocessed sequences
    manual_file_name = f"{output_file}.manual"
    # In case of translating from raw, it is convenient to just use the .manual file as input,
    # but it feels bad to read and write to it at the same time.
    if input_file == f"{output_file}.manual":
        manual_file_name = f"{output_file}.manual2"
        logger.warning(
            f"Using {manual_file_name[:-1]} as input, to prevent errors"
            f" I am renaming the *new* manual file to {manual_file_name}."
            " Make sure you keep the overview to prevent data loss."
        )
    # If .manual2 file is the input, then .manual will be overwritten again. Prevent from
    # accidently deleting precious translations.
    if input_file == f"{output_file}.manual2":
        raise ValueError(
            "Cannot use .manual2 file as input to prevent data loss. Rename file appropriately."
        )
    create_file(logger, manual_file_name, overwrite_translated)

    with open(output_file, "r+", encoding="utf-8") as f_out, open(
        manual_file_name, "r+", encoding="utf-8"
    ) as fm_out:

        ids_translated = set()
        if not overwrite_translated:
            # find examples which already are in output files
            for line in f_out:
                example = json.loads(line)
                ids_translated.add(example["id"])

            for line in fm_out:
                example = json.loads(line)
                ids_translated.add(example["id"])

            logger.info(f"Skipping {len(ids_translated)} already translated examples.")

        else:
            f_out.seek(0, 0)
            f_out.truncate()
            fm_out.seek(0, 0)
            fm_out.truncate()

        consecutive_errors = 0
        with open(input_file, "r", encoding="utf-8") as f_in:
            # accumulate sentences and send batch to google
            character_count = 0
            translate_success = 0
            translate_manual = 0
            input_count = 0
            texts = []
            examples = []
            for line in f_in:
                example = json.loads(line)
                if example["id"] in ids_translated:
                    continue
                examples.append(example)
                input_count += 1

                text = pre_process_func(example)
                texts.append(text)
                character_count += len(text)

                # send batch to google

                if len(texts) == 50 or (max_characters > 0 and character_count >= max_characters):

                    success, err, count_translated, count_manual = translate_and_write(
                        logger=logger,
                        texts=texts,
                        examples=examples,
                        tokenizer=tokenizer,
                        translation_function=translation_function,
                        source_language=source_language,
                        target_language=target_language,
                        f_out=f_out,
                        fm_out=fm_out,
                    )
                    if success:
                        consecutive_errors = 0
                        translate_success += count_translated
                        translate_manual += count_manual
                    else:
                        logger.warning(f"Warning: Error Translating: {err}")
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logger.critical("Too many consecutive Errors. Aborting.")
                            return -1
                        else:
                            logger.warning("Skipping batch.")

                    examples = []
                    texts = []
                    f_out.flush()
                    fm_out.flush()

                    if max_characters > 0 and character_count >= max_characters:
                        logger.info("Reached maximum character count. Finishing.")
                        logger.info(f"Total input examples: {input_count}")
                        logger.info(f"Translated sucessfully: {translate_success}")
                        logger.info(f"Check manually: {translate_manual}")
                        return 0

            # Need to translate potentially last batch
            if len(texts) > 0:
                success, err, count_translated, count_manual = translate_and_write(
                    logger=logger,
                    texts=texts,
                    examples=examples,
                    tokenizer=tokenizer,
                    translation_function=translation_function,
                    source_language=source_language,
                    target_language=target_language,
                    f_out=f_out,
                    fm_out=fm_out,
                )
                if not success:
                    logger.warn(f"Skipping batch, error: {err}")
                else:
                    translate_success += count_translated
                    translate_manual += count_manual

            f_out.flush()
            fm_out.flush()
            logger.info("Reached end of dataset. Finishing.")
            logger.info(f"Total input examples: {input_count}")
            logger.info(f"Translated sucessfully: {translate_success}")
            logger.info(f"Check manually: {translate_manual}")
            if len(ids_translated) + translate_success + translate_manual < input_count:
                logger.warn(
                    f"NOT ALL EXAMPLES TRANSLATED OR WRITTEN - INPUT COUNT {input_count} IS"
                    f" LARGER THAN OUTPUT COUNT ids_translated + translate_sucess + translate_manual"
                    f" {len(ids_translated) + translate_success + translate_manual}!!!"
                )
    return 0
