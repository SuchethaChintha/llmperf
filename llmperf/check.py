import sys
import argparse
from collections.abc import Iterable
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    randomly_sample_sonnet_lines_prompt,
    LLMPerfResults,
    sample_random_positive_int,
)
from tqdm import tqdm

from transformers import LlamaTokenizerFast
from box import ConfigBox
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import datetime
import yaml
from box import ConfigBox


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )
    get_token_length = lambda text: len(tokenizer.encode(text))
    timestamp=datetime.datetime.now()
    if not additional_sampling_params:
        additional_sampling_params = {}

    clients = construct_clients(llm_api=llm_api, num_clients=num_concurrent_requests)
    req_launcher = RequestsLauncher(clients)
    completed_requests = []
    num_completed_requests = 0
    start_time = time.monotonic()
    iter = 0
    pbar = tqdm(total=max_num_completed_requests)
    with open("config.json") as s:
        sub_details = ConfigBox(json.load(s))
    
    subscription_id=sub_details['subscription_id']
    resource_group=sub_details['resource_group']
    workspaces=sub_details['workspaces']
    

    while (
        time.monotonic() - start_time < test_timeout_s
        and len(completed_requests) < max_num_completed_requests
    ):
        iter += 1
        num_output_tokens = sample_random_positive_int(
            mean_output_tokens, stddev_output_tokens
        )
        # print("--num_output_tokens",num_output_tokens)

        prompt = randomly_sample_sonnet_lines_prompt(
            prompt_tokens_mean=mean_input_tokens,
            prompt_tokens_stddev=stddev_input_tokens,
            expect_output_tokens=num_output_tokens,
        )

        default_sampling_params = {"max_tokens": num_output_tokens}
        default_sampling_params.update(additional_sampling_params)
        request_config = RequestConfig(
            model=model,
            prompt=prompt,
            sampling_params=default_sampling_params,
            llm_api=llm_api,
        )
        req_launcher.launch_requests(request_config)
        # Retrieving results less frequently allows for more concurrent requests
        # to be launched. This will overall reduce the amount of time it takes
        # for the test to run.
        if not (iter % num_concurrent_requests):
            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                num_output_tokens = get_token_length(gen_text)
                request_metrics["model"]=model
                request_metrics["timestamp"]=time.monotonic()
                request_metrics["subscription_id"]=subscription_id
                request_metrics["resource_group"]=resource_group
                request_metrics["workspaces"]=workspaces
                if num_output_tokens: 
                    # print("num_output_tokens",num_output_tokens)
                    request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
                else:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                all_metrics.append(request_metrics)
            completed_requests.extend(all_metrics)
        pbar.update(len(completed_requests) - num_completed_requests)
        num_completed_requests = len(completed_requests)

    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    # check one last time that there are no remaining results to collect.
    outs = req_launcher.get_next_ready()
    # print("req_launcher",req_launcher)
    # print("outs",req_launcher.get_next_ready())
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        num_output_tokens = get_token_length(gen_text)
        print("get_token_length(gen_text)",get_token_length(gen_text))
        print("gen_text",gen_text)
        if num_output_tokens: 
            request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
        else:
            request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
        request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
        request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
        # request_metrics.append(model)        
        all_metrics.append(request_metrics)
    completed_requests.extend(all_metrics)

    print(f"\Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)
    # print("all_metrics",all_metrics)
    # print("completed_requests",completed_requests)
    print("ret",ret)

    metadata = {
        "model": model,
        "timestamp": timestamp,
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspaces": workspaces,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret
        
    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        # print("flatten item:",item)
        for sub_item in item:
            # print("subitem",sub_item)
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    # print("metrics:",metrics)
    # print("df",df)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]
    # print("REQ_OUTPUT_THROUGHPUT tokens",df_without_errored_req[
    #     common_metrics.REQ_OUTPUT_THROUGHPUT  
    # ])
    # print("output tokens",df_without_errored_req[
    #     common_metrics.NUM_OUTPUT_TOKENS
    # ])
    for key in [
        common_metrics.INTER_TOKEN_LAT,
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS
    ]:
        print("Key:",key)
        # print("df_without_errored_req[key]:",df_without_errored_req[key])
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        # print("series",series)
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        # print("quantiles.items():",quantiles.items())
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)
    
    print("output tokens",df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum())

    print("Latency",(end_time - start_time))
    overall_output_throughput = df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum() / (end_time - start_time)

    print(f"Overall Output Throughput: {overall_output_throughput}")
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min
    
    return ret


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
    """
    if mean_input_tokens < 40:
        print(
            "the minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now"
        )

    summary, individual_responses = get_token_throughput_latencies(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        mean_input_tokens=mean_input_tokens,
        stddev_input_tokens=stddev_input_tokens,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=stddev_output_tokens,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params=json.loads(additional_sampling_params),
    )

    if results_dir:
        filename = f"{model}_{mean_input_tokens}_{mean_output_tokens}"
        filename = re.sub(r"[^\w\d-]+", "-", filename)
        filename = re.sub(r"-{2,}", "-", filename)
        summary_filename = f"{filename}_summary"
        individual_responses_filename = f"{filename}_individual_responses"

        # Update to metadata.
        summary.update(user_metadata)

        results = LLMPerfResults(name=summary_filename, metadata=summary)
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        try:
            with open(results_dir / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
                json.dump(individual_responses, f, indent=4)
        except Exception as e:
            print(individual_responses)
            raise e


args = argparse.ArgumentParser(
    description="Run a token throughput and latency benchmark."
)

args.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)
args.add_argument(
    "--mean-input-tokens",
    type=int,
    default=550,
    help=(
        "The mean number of tokens to send in the prompt for the request. "
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-input-tokens",
    type=int,
    default=150,
    help=(
        "The standard deviation of number of tokens to send in the prompt for the request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--mean-output-tokens",
    type=int,
    default=150,
    help=(
        "The mean number of tokens to generate from each llm request. This is the max_tokens param "
        "for the completions API. Note that this is not always the number of tokens returned. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-output-tokens",
    type=int,
    default=80,
    help=(
        "The stdandard deviation on the number of tokens to generate per llm request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--num-concurrent-requests",
    type=int,
    default=10,
    help=("The number of concurrent requests to send (default: %(default)s)"),
)
args.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
args.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=10,
    help=(
        "The number of requests to complete before finishing the test. Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
args.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
args.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
args.add_argument(
    "--llm-api",
    type=str,
    default="openai",
    help=(
        f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--metadata",
    type=str,
    default="",
    help=(
        "A comma separated list of metadata to include in the results, e.g. "
        "name=foo,bar=1. These will be added to the metadata field of the results. "
    ),
)

def get_test_queue() -> ConfigBox:
    queue_file = f"../config.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))

if __name__ == "__main__":
    with open("conda.yaml") as y:
        config = yaml.safe_load(y)
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})
    args = args.parse_args()

    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value
    # credential = DefaultAzureCredential()
    # client = SecretClient(vault_url="https://MAAS-API.vault.azure.net", credential=credential)

    with open("Models.json") as m:
        config = json.load(m)
    for Model  in config["Models"]:
        # print(Models)
        # with open("secrets.json") as f:
        #     config = json.load(f)

        # secrets = {}
        # apikey=Models+"-apikey"
        # endpoints=Models+"-endpoint"
        # for secret_data in config["secrets"]:
        #     secret_name = secret_data["name"]
        #     # if secret_name==apikey or secret_name==endpoints:
        #         # print(secrets)
        #         # print({secrets[apikey]})
        #     try:
        #         secret = client.get_secret(secret_name)
        #         secrets[secret_name] = secret.value
        #     except (AzureKeyVaultError) as error:
        #         print(f"Error retrieving secret {secret_name}: {error}")
        # print("secrets[apikey]",secrets[apikey])
        # print("secrets[endpoint]",secrets[endpoints])
        # OPENAI_API_KEY=secrets[apikey]
        # OPENAI_API_BASE="https://Mistral-small-ouxkf-serverless.swedencentral.inference.ai.azure.com/v1"


        DT= datetime.datetime.now()
        date=DT.strftime("-%d-%m-%Y-%H-%M-%S")
        model=Model
        print("-----model",model)
        resultdir=Model+ str(date)
        print("resultdir",resultdir)
        # queue=get_test_queue()
        # run_token_benchmark(
        #     llm_api="openai",
        #     model=queue.model,
        #     test_timeout_s=600,
        #     max_num_completed_requests=queue.max_num_completed_requests,
        #     mean_input_tokens=queue.mean_input_tokens,
        #     stddev_input_tokens=queue.stddev_input_tokens,
        #     mean_output_tokens=queue.mean_output_tokens,
        #     stddev_output_tokens=queue.stddev_output_tokens,
        #     num_concurrent_requests=queue.num_concurrent_requests,
        #     additional_sampling_params='{}',
        #     results_dir=queue.results_dir,
        #     user_metadata=user_metadata,
        # )

        run_token_benchmark(
            llm_api=args.llm_api,
            model=model,
            test_timeout_s=args.timeout,
            max_num_completed_requests=args.max_num_completed_requests,
            mean_input_tokens=args.mean_input_tokens,
            stddev_input_tokens=args.stddev_input_tokens,
            mean_output_tokens=args.mean_output_tokens,
            stddev_output_tokens=args.stddev_output_tokens,
            num_concurrent_requests=args.num_concurrent_requests,
            additional_sampling_params=args.additional_sampling_params,
            results_dir=resultdir,
            user_metadata=user_metadata,
        )