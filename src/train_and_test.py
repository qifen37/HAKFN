import logging
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import datasets
import pandas as pd
from pathlib import Path

import transformers
from datasets import load_dataset
from torch.utils.data import Subset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import ExplicitEnum
from transformers.trainer_utils import get_last_checkpoint


from modeling import CodingModel, CodingModelConfig
from utils import calculate_scores, tokenize_dataset, MimicIIIDataset, \
    segment_tokenize_dataset, LazyMimicIIIDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

task_to_keys = {
    "mimic3-50": ("mimic3-50"),
    "mimic3-full": ("mimic3-full"),
}


class TransformerLayerUpdateStrategy(ExplicitEnum):
    NO = "no"
    LAST = "last"
    ALL = "all"


class DocumentPoolingStrategy(ExplicitEnum):
    FLAT = "flat"
    MAX = "max"
    MEAN = "mean"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    label_dictionary_file: Optional[str] = field(
        default=None, metadata={"help": "The name of the test data file."}
    )
    code_max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for code long titles"
        },
    )
    code_batch_size: int = field(
        default=8,
        metadata={
            "help": "The batch size for generating code representation"
        },
    )
    ignore_keys_for_eval: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys to be ignored during evaluation process."}
    )
    use_cached_datasets: bool = field(
        default=True,
        metadata={"help": "if use cached datasets to save preprocessing time. The cached datasets were preprocessed "
                          "and saved into data folder."})
    data_segmented: bool = field(
        default=False,
        metadata={"help": "if dataset is segmented or not"})

    lazy_loading: bool = field(
        default=False,
        metadata={"help": "if dataset is larger than 500MB, please use lazy_loading"})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need a training/validation file")
        elif self.label_dictionary_file is None:
            raise ValueError("label dictionary must be provided")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    # Customized model arguments
    d_model: int = field(default=768, metadata={"help": "hidden size of model. should be the same as base transformer "
                                                        "model"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout of transformer layer"})
    dropout_att: float = field(default=0.1, metadata={"help": "Dropout of label-wise attention layer"})
    num_chunks_per_document: int = field(default=0.1, metadata={"help": "Num of chunks per document"})
    transformer_layer_update_strategy: TransformerLayerUpdateStrategy = field(
        default="all",
        metadata={"help": "Update which transformer layers when training"})
    use_code_representation: bool = field(
        default=True,
        metadata={"help": "if use code representation as the "
                          "initial parameters of code vectors in attention layer"})
    multi_head_attention: bool = field(
        default=True,
        metadata={"help": "if use multi head attention for different chunks"})
    chunk_attention: bool = field(
        default=True,
        metadata={"help": "if use chunk attention for each label"})

    multi_head_chunk_attention: bool = field(
        default=True,
        metadata={"help": "if use multi head chunk attention for each label"})

    linear_init_mean: float = field(default=0.0, metadata={"help": "mean value for initializing linear layer weights"})
    linear_init_std: float = field(default=0.03, metadata={"help": "standard deviation value for initializing linear "
                                                                   "layer weights"})
    document_pooling_strategy: DocumentPoolingStrategy = field(
        default="flat",
        metadata={"help": "how to pool document representation after label-wise attention layer for each label"})


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(training_args.output_dir, "log_{}.txt".format(now)))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), file_handler],
    )

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    transformers.utils.logging.add_handler(file_handler)


    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    set_seed(training_args.seed)

    if data_args.task_name is not None and data_args.task_name not in task_to_keys.keys():
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file,
                      "label_dict": data_args.label_dictionary_file}

        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")


    label_dict = pd.read_csv(data_args.label_dictionary_file)

    num_labels = label_dict.shape[0]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        padding_side="right"
    )

    d_model = model_args.d_model

    coding_model_config = CodingModelConfig(model_args.model_name_or_path,
                                            model_args.tokenizer_name,
                                            model_args.transformer_layer_update_strategy,
                                            model_args.num_chunks_per_document,
                                            data_args.max_seq_length,
                                            model_args.dropout,
                                            model_args.dropout_att,
                                            d_model,
                                            label_dict,
                                            num_labels,
                                            model_args.use_code_representation,
                                            data_args.code_max_seq_length,
                                            data_args.code_batch_size,
                                            model_args.multi_head_attention,
                                            model_args.chunk_attention,
                                            model_args.linear_init_mean,
                                            model_args.linear_init_std,
                                            model_args.document_pooling_strategy,
                                            model_args.multi_head_chunk_attention)

    model = CodingModel(coding_model_config, training_args)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def load_data(data_file):
        cached_file = Path(data_file).parent / Path(data_file).name \
            .replace(".csv", "_seq-{}.pkl"
                     .format(data_args.max_seq_length))
        if data_args.use_cached_datasets:
            if cached_file.exists():
                with open(cached_file, "rb") as f:
                    dataset = pickle.load(f)
            else:
                data_args.use_cached_datasets = False

        if not data_args.use_cached_datasets:
            data = pd.read_csv(data_file)
            if data_args.data_segmented:
                text = data.loc[:, data.columns.str.startswith("Chunk")].fillna("").apply(
                    lambda x: [seg for seg in x],
                    axis=1).tolist()
                labels = data.iloc[:, 11:61].apply(lambda x: [seg for seg in x], axis=1).tolist()
                leaves = data.iloc[:, 61:].apply(lambda x: [seg for seg in x], axis=1).tolist()
                results = tokenize_dataset(tokenizer, text, labels, leaves, data_args.max_seq_length)
            else:
                text = data["text"].tolist()
                import ast
                labels = data["labels"].apply(ast.literal_eval).tolist()

                results = segment_tokenize_dataset(tokenizer, text, labels,
                                                   data_args.max_seq_length,
                                                   model_args.num_chunks_per_document)

            dataset = MimicIIIDataset(results)
            with open(cached_file, 'wb') as f:
                pickle.dump(dataset, f)

        return dataset

    if training_args.do_train:
        if data_args.lazy_loading:
            train_dataset = LazyMimicIIIDataset(data_args.train_file, data_args.task_name, 'train')
        else:
            train_dataset = load_data(data_args.train_file)
        if data_args.max_train_samples is not None:
            train_dataset = Subset(train_dataset, list(range(data_args.max_train_samples)))

    if training_args.do_eval:
        if data_args.lazy_loading:
            eval_dataset = LazyMimicIIIDataset(data_args.validation_file, data_args.task_name, 'dev')
        else:
            eval_dataset = load_data(data_args.validation_file)
        if data_args.max_eval_samples is not None:
            eval_dataset = Subset(eval_dataset, list(range(data_args.max_eval_samples)))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if data_args.lazy_loading:
            predict_dataset = LazyMimicIIIDataset(data_args.test_file, data_args.task_name, 'test')
        else:
            predict_dataset = load_data(data_args.test_file)
        if data_args.max_predict_samples is not None:
            predict_dataset = Subset(predict_dataset, list(range(data_args.max_predict_samples)))

    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        metric_scores = calculate_scores(p.label_ids, logits)
        micro_scores = calculate_scores(p.label_ids, logits, average="micro")
        metric_scores.update(micro_scores)

        return metric_scores

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint,
                                     ignore_keys_for_eval=data_args.ignore_keys_for_eval)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:

        metrics = trainer.evaluate(eval_dataset=eval_dataset, ignore_keys=data_args.ignore_keys_for_eval)
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:

        output = trainer.predict(predict_dataset, metric_key_prefix="predict")

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{data_args.task_name}.pkl")
        predict_result = {"labels": output.label_ids, "metrics": output.metrics, "predictions": output.predictions}
        if trainer.is_world_process_zero():
            with open(output_predict_file, "wb") as writer:
                pickle.dump(predict_result, writer, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
