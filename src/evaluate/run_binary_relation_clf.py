# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script extends Sherlock's run_binary_relation_clf.py with a simple mechanism to include additional files for
the train and dev splits. Used to support MultiTACRED's multi-language experiments with English + "sample of target
language data" for training.
"""


from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import random
import shutil
import sys
import traceback
from typing import Optional, Sequence

import numpy as np
import torch
from numpy.random import choice
from sherlock.dataset import TensorDictDataset
from sherlock.dataset_readers import DatasetReader
from sherlock.feature_converters import FeatureConverter
from sherlock.metrics import compute_f1
from sherlock.tasks import IETask
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForSequenceClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "camembert": (CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer),
}

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, dataset_reader, converter, model, tokenizer):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = load_and_cache_examples(
        args, dataset_reader, converter, tokenizer, split="train"
    )
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss, logging_loss2 = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        preds = []
        out_label_ids = []
        instance_ids = []
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if "metadata" in batch:
                instance_ids.append(batch["metadata"]["guid"])
                del batch["metadata"]
            batch = {k: t.to(args.device) for k, t in batch.items()}

            outputs = model(**batch)
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)

            preds.append(logits.detach().cpu().numpy())
            out_label_ids.append(batch["labels"].detach().cpu().numpy())

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 and not args.tpu:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _, _ = evaluate(
                            args, dataset_reader, converter, model, tokenizer, split="dev"
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("Steps/Eval/{}".format(key), value, global_step)
                    tb_writer.add_scalar("Steps/lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "Steps/Train/loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.tpu:
                args.xla_model.optimizer_step(optimizer, barrier=True)
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # Evaluate at end of epoch
        if args.local_rank == -1:
            # Only evaluate when single GPU otherwise metrics may not average well

            preds = np.concatenate(preds, axis=0)
            preds = np.argmax(preds, axis=1)
            out_label_ids = np.concatenate(out_label_ids, axis=0)
            instance_ids = np.concatenate(
                instance_ids, axis=0
            )  # currently not used, but if we want to save train preds
            train_results = compute_f1(preds, out_label_ids)
            for key, value in train_results.items():
                tb_writer.add_scalar(f"Epoch/Train/{key}", value, epoch)

            test_results, _, _, _ = evaluate(
                args,
                dataset_reader,
                converter,
                model,
                tokenizer,
                split="dev",
            )

            for key, value in test_results.items():
                tb_writer.add_scalar(f"Epoch/Eval/{key}", value, epoch)

            # Save results to file
            log_file_dir = os.path.join(args.output_dir, f"metrics_epoch_{epoch}.json")
            results = {f"training_{k}": v for k, v in train_results.items()}
            results.update({f"validation_{k}": v for k, v in test_results.items()})
            with open(log_file_dir, "w", encoding="utf-8") as result_out:
                json.dump(results, result_out, indent=4)

        # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar(
            "Train/Epoch/loss", (tr_loss - logging_loss2) / len(train_dataloader), epoch
        )
        logging_loss2 = tr_loss

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(
    args, dataset_reader, converter, model, tokenizer, split, filename="eval_results.txt"
):
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, dataset_reader, converter, tokenizer, split)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # Eval!
    name = filename or ""
    logger.info(f"***** Running evaluation {name} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    true_label_ids = []
    instance_ids = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        if "metadata" in batch:
            instance_ids.append(batch["metadata"]["guid"])
            del batch["metadata"]
        batch = {k: t.to(args.device) for k, t in batch.items()}
        if args.model_type not in ["bert", "xlnet"]:
            batch["token_type_ids"] = None

        if args.model_type == "distilbert":
            del batch["token_type_ids"]

        with torch.no_grad():
            outputs = model(**batch)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        preds.append(logits.detach().cpu().numpy())
        true_label_ids.append(batch["labels"].detach().cpu().numpy())

    eval_loss = eval_loss / nb_eval_steps
    preds = np.concatenate(preds, axis=0)
    preds = np.argmax(preds, axis=1)
    true_label_ids = np.concatenate(true_label_ids, axis=0)
    instance_ids = np.concatenate(instance_ids, axis=0)
    result = compute_f1(preds, true_label_ids)

    logger.info("***** Eval results {} *****".format(name))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    if filename is not None:
        output_eval_file = os.path.join(eval_output_dir, filename)
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result, preds, true_label_ids, instance_ids


def load_and_cache_examples(args, dataset_reader, converter, tokenizer, split):
    if args.local_rank not in [-1, 0] and split not in ["dev", "test"]:
        # Make sure only the first process in distributed training process
        # the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.cache_dir,
        "cached_rc_{}_{}_{}".format(
            split,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info(
            "Loading features for split %s from cached file %s", split, cached_features_file
        )
        input_features = torch.load(cached_features_file)
    else:

        additional_input_files = []
        if split == "train":
            file_path = os.path.join(args.data_dir, args.train_file)
            if args.additional_train_files:
                additional_input_files = args.additional_train_files.split(",")
        elif split == "dev":
            file_path = os.path.join(args.data_dir, args.dev_file)
            if args.additional_dev_files:
                additional_input_files = args.additional_dev_files.split(",")
        elif split == "test":
            file_path = os.path.join(args.data_dir, args.test_file)

        logger.info("Creating features for split %s from dataset file at %s", split, file_path)
        documents = list(dataset_reader.get_documents(file_path))

        for additional_input_file in additional_input_files:
            file_path = os.path.join(args.data_dir, additional_input_file)
            logger.info(
                f"Adding features for split {split} from additional dataset file at {file_path} "
                f"(sample size = {args.additional_train_dev_files_sample_size}, "
                f"stratified = {args.additional_train_dev_files_use_stratified_sampling})"
            )
            additional_docs = list(dataset_reader.get_documents(file_path))
            additional_docs_labels = [d.rels[0].label for d in additional_docs]
            try:
                train_test = train_test_split(
                    additional_docs,
                    train_size=args.additional_train_dev_files_sample_size,
                    random_state=args.seed,
                    stratify=additional_docs_labels,
                )
                documents.extend(train_test[0])
                logger.info(f"Added {len(train_test[0])} documents to {split}.")
            except Exception:
                try:
                    # raised if too few members per class in stratified split
                    train_test = train_test_split(
                        additional_docs,
                        train_size=args.additional_train_dev_files_sample_size,
                        random_state=args.seed,
                    )
                    documents.extend(train_test[0])
                    logger.warning(
                        f"Added {len(train_test[0])} documents to {split}, NOT stratified."
                    )
                except Exception:
                    traceback.print_exc()
                    logger.warning(
                        f"Using random {args.additional_train_dev_files_sample_size}% documents"
                    )
                    sample_number = min(
                        int(args.additional_train_dev_files_sample_size * len(additional_docs)),
                        len(additional_docs),
                    )
                    documents.extend(choice(additional_docs, sample_number, replace=False))
                    logger.info(f"Added {sample_number} documents to {split}, NOT stratified.")

        input_features = converter.documents_to_features(documents)

        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features for split %s into cached file %s", split, cached_features_file
            )
            os.makedirs(args.cache_dir, exist_ok=True)
            torch.save(input_features, cached_features_file)

    if args.local_rank == 0 and split not in ["dev", "test"]:
        torch.distributed.barrier()

    tensor_dicts = []
    for features in input_features:
        tensor_dict = {
            "input_ids": torch.tensor(features.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(features.attention_mask, dtype=torch.long),
            "metadata": features.metadata,
        }
        if features.token_type_ids is not None:
            tensor_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.labels is not None:
            tensor_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        tensor_dicts.append(tensor_dict)

    dataset = TensorDictDataset(tensor_dicts)
    return dataset


def main(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the hub "
        + "https://huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Task-specific parameters
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument(
        "--entity_handling",
        type=str,
        default="mask_entity",
        choices=[
            "mark_entity",
            "mark_entity_append_ner",
            "mask_entity",
            "mask_entity_append_text",
        ],
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run predictions on the test set."
    )
    parser.add_argument(
        "--add_inverse_relations",
        action="store_true",
        help="Whether to also add inverse relations to the document.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="Where do you want to store the pre-trained models downloaded from s3 and cached features",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=50, help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Whether to run on the TPU defined in the environment variables",
    )
    parser.add_argument(
        "--tpu_ip_address",
        type=str,
        default="",
        help="TPU IP address if none are set in the environment variables",
    )
    parser.add_argument(
        "--tpu_name",
        type=str,
        default="",
        help="TPU name if none are set in the environment variables",
    )
    parser.add_argument(
        "--xrt_tpu_config",
        type=str,
        default="",
        help="XRT TPU config if none are set in the environment variables",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--max_instances",
        type=int,
        default=-1,
        help="Only use this number of first instances in dataset (e.g. for debugging).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.json",
        help="Train file name relative to --data_dir",
    )
    parser.add_argument(
        "--dev_file", type=str, default="dev.json", help="Dev file name relative to --data_dir"
    )
    parser.add_argument(
        "--test_file", type=str, default="test.json", help="Test file name relative to --data_dir"
    )
    parser.add_argument(
        "--dataset_reader",
        type=str,
        default="tacred",
        choices=["tacred", "tacred_dfki_jsonl"],
        help="Registered dataset reader name from ['tacred', 'tacred_dfki_jsonl']",
    )
    parser.add_argument(
        "--predictions_exp_name",
        type=str,
        default="",
        help="Optional experiment name identifier which is appended to the prediction file name",
    )

    parser.add_argument(
        "--additional_train_files",
        type=str,
        default="",
        help="Comma-separated list of additional train file names, relative to --data_dir",
    )
    parser.add_argument(
        "--additional_dev_files",
        type=str,
        default="",
        help="Comma-separated list of additional dev file names, relative to --data_dir",
    )
    parser.add_argument(
        "--additional_train_dev_files_sample_size",
        type=float,
        default=0.1,
        help="Proportion of each additional train/dev file to include during training, must be between 0.0 and 1.0",
    )
    parser.add_argument(
        "--additional_train_dev_files_use_stratified_sampling",
        action="store_true",
        help="If true, use stratified sampling to obtain instances from additional train/dev files",
    )

    args = parser.parse_args(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        stream=sys.stdout,
    )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    elif os.path.exists(args.output_dir) and args.do_train:
        logger.warning(f"Deleting content of output_dir: {args.output_dir}")
        # delete all files in old dir
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    elif args.do_train:
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    if args.tpu:
        if args.tpu_ip_address:
            os.environ["TPU_IP_ADDRESS"] = args.tpu_ip_address
        if args.tpu_name:
            os.environ["TPU_NAME"] = args.tpu_name
        if args.xrt_tpu_config:
            os.environ["XRT_TPU_CONFIG"] = args.xrt_tpu_config

        assert "TPU_IP_ADDRESS" in os.environ
        assert "TPU_NAME" in os.environ
        assert "XRT_TPU_CONFIG" in os.environ

        import torch_xla  # noqa: F401
        import torch_xla.core.xla_model as xm

        args.device = xm.xla_device()
        args.xla_model = xm

    # Setup logging
    # logging.basicConfig(
    #    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #    datefmt="%m/%d/%Y %H:%M:%S",
    #    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    #    stream=sys.stdout,
    # )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    DatasetReaderClass = DatasetReader.by_name(args.dataset_reader)

    dataset_reader = DatasetReaderClass(
        add_inverse_relations=args.add_inverse_relations,
        negative_label_re=args.negative_label,
        max_instances=args.max_instances if args.max_instances != -1 else None,
    )

    train_path = os.path.join(args.data_dir, args.train_file)
    labels = dataset_reader.get_labels(IETask.BINARY_RC, train_path)
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels
    )
    logger.info(f"Initialized config class {config.__class__}: {config}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        # set to False because Sherlock's binary_rc FeatureConverter currently fails in encode_plus call
        # due to wrong input type (line 206). See also Line 933, 966
        use_fast=False,
    )
    logger.info(f"Initialized tokenizer class {tokenizer.__class__}: {tokenizer}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config
    )
    logger.info(f"Initialized model class {model.__class__}: {model}")

    BinaryRcConverter = FeatureConverter.by_name("binary_rc")

    converter = BinaryRcConverter(
        labels=labels,
        max_length=args.max_seq_length,
        tokenizer=tokenizer,
        entity_handling=args.entity_handling,
        log_num_input_features=3,
    )

    additional_tokens = dataset_reader.get_additional_tokens(IETask.BINARY_RC, train_path)

    if additional_tokens:
        tokenizer.add_tokens(additional_tokens)
        model.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, dataset_reader, converter, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if (
        args.do_train
        and (args.local_rank == -1 or torch.distributed.get_rank() == 0)
        and not args.tpu
    ):

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        converter.save(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir,
            do_lower_case=args.do_lower_case,
            # set to False because Sherlock's binary_rc FeatureConverter currently fails in encode_plus call
            # due to wrong input type (line 206). See also Line 862, 966
            use_fast=False,
        )
        converter = BinaryRcConverter.from_pretrained(args.output_dir, tokenizer=tokenizer)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            filename = f"{prefix}_eval_result.txt"

            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result, _, _, _ = evaluate(
                args, dataset_reader, converter, model, tokenizer, split="dev", filename=filename
            )
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_predict and args.local_rank in [-1, 0]:
        if args.do_train:
            # this reloading fails if do_train = False, therefore - only reload if do_train = True
            tokenizer = AutoTokenizer.from_pretrained(
                args.output_dir,
                do_lower_case=args.do_lower_case,
                # set to False because binary_rc FeatureConverter currently fails in encode_plus call
                # due to wrong input type (line 206). See also lines 862, 933
                use_fast=False,
            )
            logger.info(f"Initialized tokenizer class {tokenizer.__class__}: {tokenizer}")
            model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
            logger.info(f"Initialized model class {model.__class__}: {model}")
            model.to(args.device)
        result, predictions, true_label_ids, instance_ids = evaluate(
            args, dataset_reader, converter, model, tokenizer, split="test", filename=None
        )
        predictions = [
            {
                "id": instance_ids[idx],
                "label_true": converter.id_to_label_map[true_label_ids[idx]],
                "label_pred": converter.id_to_label_map[i],
            }
            for idx, i in enumerate(predictions)
        ]

        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.jsonl")

        with open(output_test_predictions_file, "w") as writer:
            for prediction in predictions:
                writer.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        results.update(result)

    return results


if __name__ == "__main__":
    main()
