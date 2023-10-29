import os
import time
from math import ceil
from typing import Optional, Callable, Dict

import datasets
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from src import MODELS_DIR, ExperimentConfig
from src.evaluate.evaluator import RecEvaluator
from src.utils import log_wandb
from src.data.datasets.amazon_dataset import AmazonDataset
from src.data.templates.templates import Task
from src.evaluate.metrics import Metric
from src.model.t5 import T5Rec


class RecTrainer:

    def __init__(self,
                 n_epochs: int,
                 batch_size: int,
                 rec_model,
                 train_sampling_fn: Callable[[Dict], Dict],
                 device: str = 'cuda:0',
                 monitor_metric: str = 'loss',
                 eval_batch_size: Optional[int] = None,
                 output_name: Optional[str] = None,
                 random_seed: Optional[int] = None):

        self.rec_model = rec_model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_sampling_fn = train_sampling_fn
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.device = device
        self.random_seed = random_seed
        self.monitor_metric = monitor_metric

        # output name
        if output_name is None:
            # replace '/' with '_' to avoid creation of subdir (google/flan-t5-small -> google_flan-t5-small)
            output_name = f"{rec_model.config.name_or_path.replace('/', '_')}_{n_epochs}"

        self.output_name = output_name
        self.output_path = os.path.join(MODELS_DIR, output_name)

        # evaluator for validating with validation set during training
        self.rec_evaluator = RecEvaluator(self.rec_model, self.eval_batch_size)

        # from strings to objects initialized
        train_task_list = Task.from_string(*rec_model.config.training_tasks_str, all_unique_items=all_labels)

        # Log all templates used
        dataframe_dict = {"task_type": [], "template_id": [], "input_prompt": [], "target_text": []}
        for task in train_task_list:
            for template_id in task.templates_dict:
                input_prompt, target_text = task.templates_dict[template_id]

                dataframe_dict["task_type"].append(str(task))
                dataframe_dict["template_id"].append(template_id)
                dataframe_dict["input_prompt"].append(input_prompt)
                dataframe_dict["target_text"].append(target_text)

        log_wandb({"task_templates": wandb.Table(dataframe=pd.DataFrame(dataframe_dict))})

    def train(self, train_dataset: datasets.Dataset, validation_dataset: datasets.Dataset = None):

        self.rec_model.train()

        # init variables for saving best model thanks to validation set (if present)
        best_epoch = -1
        best_res_op_comparison = None
        best_val_monitor_result = None

        # depending on the monitor metric, in order to find the best model we should either
        # minimize the metric (e.g. loss) or maximize it (e.g. hit)
        if validation_dataset is not None:
            [monitor_metric_obj] = Metric.from_string(self.monitor_metric)
            best_res_op_comparison = monitor_metric_obj.operator_comparison

            # small trick to get the initialization value
            best_val_monitor_result = +np.inf if best_res_op_comparison(-np.inf, +np.inf) else -np.inf

        optimizer = self.rec_model.get_suggested_optimizer()

        start = time.time()
        for current_epoch in range(start=1, stop=self.n_epochs + 1):

            self.rec_model.train()

            # at the start of each iteration, we randomly sample the train sequence and tokenize it
            # batched set to True because data can be augmented, either when sampling or when
            # tokenizing (e.g. a task as multiple support templates)

            sampled_train = train_dataset.map(self.train_sampling_fn,
                                              remove_columns=train_dataset.column_names,
                                              keep_in_memory=True,
                                              batched=True,
                                              desc="Sampling train set")

            preprocessed_train = sampled_train.map(self.rec_model.tokenize,
                                                   remove_columns=sampled_train.column_names,
                                                   keep_in_memory=True,
                                                   batched=True,
                                                   desc="Tokenizing train set")

            # shuffle here so that if we augment data (2 or more row for a single user) it is shuffled
            preprocessed_train = preprocessed_train.shuffle()
            preprocessed_train.set_format("torch")

            # ceil because we don't drop the last batch
            total_n_batch = ceil(preprocessed_train.num_rows / self.batch_size)

            pbar = tqdm(preprocessed_train.iter(batch_size=self.batch_size),
                        total=total_n_batch)

            train_loss = 0

            # progress will go from 0 to 100. Init to -1 so at 0 we perform the first print
            progress = -1
            for i, batch in enumerate(pbar, start=1):

                optimizer.zero_grad()

                prepared_input = self.rec_model.prepare_input(batch)
                loss = self.rec_model.train_step(prepared_input)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # we update the loss every 1% progress considering the total n° of batches.
                # tqdm update integer percentage (1%, 2%) when float percentage is over .5 threshold (1.501 -> 2%)
                # so we print infos in the same way
                if round(100 * (i / total_n_batch)) > progress:
                    pbar.set_description(f"Epoch {current_epoch}, Loss -> {(train_loss / i):.6f}")
                    progress += 1
                    log_wandb({
                        "train/loss": train_loss / i
                    })

            train_loss /= total_n_batch

            pbar.close()

            dict_to_log = {
                "train/loss": train_loss,
                "train/epoch": current_epoch
            }

            if validation_dataset is not None:

                self.rec_model.eval()

                # we surely want loss for the progbar
                metric_list_str = ["loss"]
                if self.monitor_metric != "loss":
                    metric_list_str.append(self.monitor_metric)

                val_result = self.rec_evaluator.evaluate(
                    validation_dataset,
                    metric_list_str=metric_list_str,
                    return_loss=True)

                monitor_val = val_result[self.monitor_metric]
                should_save = best_res_op_comparison(monitor_val,
                                                     best_val_monitor_result)

                # we save the best model based on the metric/loss result
                if should_save:
                    best_epoch = current_epoch  # we start from 0
                    best_val_monitor_result = monitor_val
                    self.rec_model.save_pretrained(self.output_path)

                    print(f"Validation {self.monitor_metric} improved, model saved into {self.output_path}!")

                # prefix "val" for val result dict
                val_to_log = {f"val/{metric_name}": metric_val for metric_name, metric_val in val_result.items()}
                val_to_log["val/epoch"] = current_epoch

                dict_to_log.update(val_to_log)

            # if no validation set, we simply save the model of the last epoch
            elif current_epoch == self.n_epochs:
                self.rec_model.save_pretrained(self.output_path)

            # log to wandb at each epoch
            log_wandb(dict_to_log)

        elapsed_time = (time.time() - start) / 60
        print(" Train completed! Check models saved into 'models' dir ".center(100, "*"))
        print(f"Time -> {elapsed_time}")
        print(f"Best epoch -> {best_epoch}")

        log_wandb({
            "train/elapsed_time": elapsed_time,
            "train/best_epoch": best_epoch
        })

        # return best model path if validation was set, otherwise this return the path of the model
        # saved at the last epoch
        return self.output_path


def trainer_main():
    exp_name = ExperimentConfig.exp_name
    n_epochs = ExperimentConfig.n_epochs
    batch_size = ExperimentConfig.train_batch_size
    eval_batch_size = ExperimentConfig.eval_batch_size
    device = ExperimentConfig.device
    checkpoint = ExperimentConfig.checkpoint
    random_seed = ExperimentConfig.random_seed
    train_tasks = ExperimentConfig.train_tasks
    monitor_metric = ExperimentConfig.monitor_strategy
    inject_personalization = ExperimentConfig.inject_personalization
    val_task = ExperimentConfig.val_task
    val_task_template_id = ExperimentConfig.val_task_template_id

    ds = AmazonDataset.load()

    ds_dict = ds.get_hf_datasets()
    all_unique_labels = ds.all_items
    all_unique_users = ds.all_users
    sampling_fn = ds.sample_train_sequence

    train = ds_dict["train"]
    val = ds_dict["validation"] if monitor_metric != "no" else None

    # REDUCE FOR TESTING
    # train = Dataset.from_dict(train[:100])
    # val = Dataset.from_dict(val[:100])

    rec_model = T5Rec.from_pretrained(
        checkpoint,
        training_tasks_str=train_tasks,
        n_users=len(all_unique_users),
        all_unique_labels=all_unique_labels.tolist(),
        eval_task_str=val_task,
        force_eval_template_id=val_task_template_id,
        inject_personalization=inject_personalization,
        device=device,
    )

    trainer = RecTrainer(
        rec_model=rec_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        random_seed=random_seed,
        train_sampling_fn=sampling_fn,
        monitor_metric=monitor_metric,
        output_name=exp_name
    )

    trainer.train(train, validation_dataset=val)


if __name__ == "__main__":
    ExperimentConfig.content_indexing = True
    ExperimentConfig.inject_personalization = ("train", "eval")
    ExperimentConfig.train_tasks = ("SequentialSideInfoTask",)

    trainer_main()
