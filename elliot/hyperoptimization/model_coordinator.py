"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import datetime
from types import SimpleNamespace
import typing as t
import numpy as np
import logging as pylog
import time
import pandas as pd
from elliot.utils import logging
import os
from hyperopt import STATUS_OK
from datetime import datetime


class ModelCoordinator(object):
    """
    This class handles the selection of hyperparameters for the hyperparameter tuning realized with HyperOpt.
    """

    def __init__(self, data_objs, base: SimpleNamespace, params, model_class: t.ClassVar, test_fold_index: int):
        """
        The constructor creates a Placeholder of the recommender model.

        :param base: a SimpleNamespace that contains the configuration (main level) options
        :param params: a SimpleNamespace that contains the hyper-parameters of the model
        :param model_class: the class of the recommendation model
        """
        self.logger = logging.get_logger(self.__class__.__name__, pylog.CRITICAL if base.config_test else pylog.DEBUG)
        self.data_objs = data_objs
        self.base = base
        self.params = params
        self.model_class = model_class
        self.test_fold_index = test_fold_index
        self.model_config_index = 0

    def objective(self, args):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """

        sampled_namespace = SimpleNamespace(**args)
        model_params = SimpleNamespace(**self.params[0].__dict__)

        self.logger.info("Hyperparameter tuning exploration:")
        for (k, v) in sampled_namespace.__dict__.items():
            model_params.__setattr__(k, v)
            self.logger.info(f"Exploration for {k}. Value extracted: {model_params.__getattribute__(k)}")

        losses = []
        results = []
        times = []
        for trainval_index, data_obj in enumerate(self.data_objs):
            self.logger.info(f"Exploration: Hyperparameter exploration number {self.model_config_index + 1}")
            self.logger.info(f"Exploration: Test Fold exploration number {self.test_fold_index + 1}")
            self.logger.info(f"Exploration: Train-Validation Fold exploration number {trainval_index + 1}")
            model = self.model_class(data=data_obj, config=self.base, params=model_params)
            tic = time.perf_counter()
            # N.B Per adesso solo UserKNN e ItemKNN ritornano qualcosa dalla train function
            whole_similarity_time = model.train()
            toc = time.perf_counter()
            times.append(toc - tic)
            losses.append(model.get_loss())
            results.append(model.get_results())
            if whole_similarity_time is not None:
                # Chiedere se ha senso aggiungerlo qui.
                results[-1][self.base.evaluation.cutoffs]["val_results"].update(
                    {"similarity_time": whole_similarity_time})
                results[-1][self.base.evaluation.cutoffs]["test_results"].update(
                    {"similarity_time": whole_similarity_time})
            if getattr(model._model, "_lsh_times_obj", None):
                results[-1][self.base.evaluation.cutoffs]["val_results"].update(model._model._lsh_times_obj)
                results[-1][self.base.evaluation.cutoffs]["test_results"].update(model._model._lsh_times_obj)

        self.model_config_index += 1

        loss = np.average(losses)
        results_mean = self._average_results(results)
        if model_params.similarity in ["rp_faiss", "rp_custom", "rp_hashtables"]:
            print_baseline_comparisons(self.base, model_params, results_mean)
        # save_running_results(results_mean, model_params, self.base)

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': model.get_params(),
            'val_results': {k: result_dict["val_results"] for k, result_dict in results_mean.items()},
            # 'val_std_results': {k: result_dict["val_results"] for k, result_dict in results_std.items()},
            'val_statistical_results': {k: result_dict["val_statistical_results"] for k, result_dict in
                                        model.get_results().items()},
            'test_results': {k: result_dict["test_results"] for k, result_dict in results_mean.items()},
            # 'test_std_results': {k: result_dict["test_results"] for k, result_dict in results_std.items()},
            'test_statistical_results': {k: result_dict["test_statistical_results"] for k, result_dict in
                                         model.get_results().items()},
            'time': times,
            'name': model.name
        }

    def single(self):
        """
        This function respect the signature, and the return format required for HyperOpt optimization
        :param args: a Dictionary that contains the new hyper-parameter values that will be used in the current run
        :return: it returns a Dictionary with loss, and status being required by HyperOpt,
        and params, and results being required by the framework
        """

        self.logger.info("Hyperparameters:")
        for k, v in self.params.__dict__.items():
            self.logger.info(f"{k} set to {v}")
        # TODO: Aggiungere anche qui la logica di objective

        losses = []
        results = []
        times = []
        for trainval_index, data_obj in enumerate(self.data_objs):
            self.logger.info(f"Exploration: Test Fold exploration number {self.test_fold_index + 1}")
            self.logger.info(f"Exploration: Train-Validation Fold exploration number {trainval_index + 1}")
            model = self.model_class(data=data_obj, config=self.base, params=self.params)
            tic = time.perf_counter()
            model.train()
            toc = time.perf_counter()
            times.append(toc - tic)
            losses.append(model.get_loss())
            results.append(model.get_results())

        loss = np.average(losses)
        results_mean = self._average_results(results)
        # results_std = self._std_results(results)

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': model.get_params(),
            'val_results': {k: result_dict["val_results"] for k, result_dict in results_mean.items()},
            # 'val_std_results': {k: result_dict["val_results"] for k, result_dict in results_std.items()},
            'val_statistical_results': {k: result_dict["val_statistical_results"] for k, result_dict in
                                        model.get_results().items()},
            'test_results': {k: result_dict["test_results"] for k, result_dict in results_mean.items()},
            # 'test_std_results': {k: result_dict["test_results"] for k, result_dict in results_std.items()},
            'test_statistical_results': {k: result_dict["test_statistical_results"] for k, result_dict in
                                         model.get_results().items()},
            'time': times,
            'name': model.name
        }

    @staticmethod
    def _average_results(results_list):
        ks = list(results_list[0].keys())
        eval_result_types = ["val_results", "test_results"]
        metrics = list(results_list[0][ks[0]]["val_results"].keys())
        return {k: {type_: {metric: np.average([fold_result[k][type_][metric]
                                                for fold_result in results_list])
                            for metric in metrics}
                    for type_ in eval_result_types}
                for k in ks}

    @staticmethod
    def _std_results(results_list):
        ks = list(results_list[0].keys())
        eval_result_types = ["val_results"]
        metrics = list(results_list[0][ks[0]]["val_results"].keys())
        return {k: {type_: {metric: np.std([fold_result[k][type_][metric] for fold_result in results_list])
                            for metric in metrics}
                    for type_ in eval_result_types}
                for k in ks}


def save_running_results(results_mean, model_config, base):
    """
    TO BE CHECKED IF HELPFUL
    :param results_mean:
    :param model_config:
    :param base:
    :return:
    """
    print("Saving Experiment results")
    test_results_dict = results_mean[base.evaluation.cutoffs]["test_results"]
    test_results_dict["model"] = model_config.name
    output_df = pd.DataFrame.from_dict(
        {k: [v] for k, v in test_results_dict.items()})
    output_df.to_csv(sep="\t", index=False)


def print_baseline_comparisons(base, model_params, results_mean):
    print("------Experiments vs Baseline------------------")
    test_results_dict = results_mean[base.evaluation.cutoffs]["test_results"].copy()
    file_path = os.path.join("results_lsh", "comparisons", base.dataset, model_params.name.split("_")[0],
                             model_params.name.split("_")[0] + "_" + "baseline" + ".tsv")

    if os.path.exists(file_path):
        print("We have a baseline")
        baseline_df = pd.read_csv(file_path, sep="\t")
        ndgc_loss = ((baseline_df["nDCGRendle2020"][0] - test_results_dict["nDCGRendle2020"]) /
                     baseline_df["nDCGRendle2020"][0]) * 100
        similarity_time_change = ((baseline_df["similarity_time"][0] - test_results_dict["similarity_time"]) /
                                  baseline_df["similarity_time"][0]) * 100
        model_name = model_params.name.split("_")[0]
        print("--------------------CURRENT EXPERIMENT STATS-------------------------------")
        print(f" NDCG-Loss: {ndgc_loss} %")
        print(f" similarity_time_change:  {similarity_time_change} %")
        print(f" nbits: {model_params.nbits}")
        print(f" ntables: {model_params.ntables}")
        print("-----------------------------------------------------------------------------")
        log_path = f"lsh_logs/{base.dataset}/{model_name}"
        os.makedirs(log_path, exist_ok=True)
        test_results_dict["dataset"] = base.dataset
        test_results_dict["nbits"] = model_params.nbits
        test_results_dict["ntables"] = model_params.ntables
        test_results_dict["model"] = model_name + ":" + model_params.similarity
        print("Saving Results to ")
        output_df = pd.DataFrame.from_dict(
            {k: [v] for k, v in test_results_dict.items()})
        output_df.to_csv(f"{log_path}/{test_results_dict['model']}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.tsv")
    else:
        print("File does not exist")
