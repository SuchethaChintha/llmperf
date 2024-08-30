import json
import os
import time
import dotenv
from typing import Any, Dict
import ray
import requests
from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics
from azure.identity import ManagedIdentityCredential,DefaultAzureCredential,ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv

load_dotenv()

@ray.remote
class OpenAIChatCompletionsClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
        # credential = DefaultAzureCredential()
        # credential = ManagedIdentityCredential()
        # client_id=os.getenv("client_id")
        # client_secret=os.getenv("client_secret")
        # tenant_id=os.getenv("tenant_id")
        # credential = ClientSecretCredential(client_id=client_id, client_secret=client_secret,tenant_id=tenant_id)
        managed_identity_client_id=os.getenv("client_id")
        credential = ManagedIdentityCredential(client_id = managed_identity_client_id)
        client = SecretClient(vault_url="https://Llmperf.vault.azure.net", credential=credential)
        with open("Models.json") as m:
            items = json.load(m)
        model_list=items["Models"]
        secret_name=items["secrets"]
        
        if model in model_list:
            secrets = {}
            apikey=model+"-apikey-westus"
            endpoints=model+"-endpoint-westus"
            apiversion=model+"-apiversion-westus"
            for secret_data in secret_name:
                secret_name = secret_data["name"]
            # if secret_name==apikey or secret_name==endpoints:
                # print(secrets)
                # print({secrets[apikey]})
                try:
                    secret = client.get_secret(secret_name)
                    secrets[secret_name] = secret.value
                except (AzureKeyVaultError) as error:
                    print(f"Error retrieving secret {secret_name}: {error}")
            
            key= secrets[apikey]
            headers = {"Authorization": f"Bearer {key}"}
            # address= secrets[endpoints]
            address= secrets[endpoints]+'/chat/completions' 
            if "openai.azure.com" in address:
                api_version = secrets[apiversion]
                if not api_version:
                    raise ValueError("the environment variable OPENAI_API_VERSION must be set for Azure OpenAI service.")
                address = f"{address}?api-version={api_version}"
                headers = {"api-key": key}  # replace with Authorization: Bearer       
            print("address",address)
            if "azure" in address:
                azure="Azure MaaS"
            else:
                azure="not azure"
            # if not address.endswith("/"):
                # address = address + "/"
            # address += "chat/completions"
            body_content=json.dumps(body, indent=2)
            body_response=""
            request_id=""
            try:
                with requests.post(
                    address,
                    json=body,
                    stream=True,
                    timeout=180,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_msg = response.text
                        error_response_code = response.status_code
                        response.raise_for_status()
                    request_id = response.headers["x-request-id"]    
                    for chunk in response.iter_lines(chunk_size=None):
                        chunk = chunk.strip()

                        if not chunk:
                            continue
                        stem = "data: "
                        chunk = chunk[len(stem) :]
                        if chunk == b"[DONE]":
                            continue
                        tokens_received += 1
                        data = json.loads(chunk)

                        if "error" in data:
                            error_msg = data["error"]["message"]
                            error_response_code = data["error"]["code"]
                            raise RuntimeError(data["error"]["message"])
                            
                        delta = data["choices"][0]["delta"]
                        if delta.get("content", None):
                            if not ttft:
                                ttft = time.monotonic() - start_time
                                time_to_next_token.append(ttft)
                            else:
                                time_to_next_token.append(
                                    time.monotonic() - most_recent_received_token_time
                                )
                            most_recent_received_token_time = time.monotonic()
                            generated_text += delta["content"]
                            body_response=generated_text

                total_request_time = time.monotonic() - start_time
                output_throughput = tokens_received / total_request_time
            except Exception as e:
                metrics[common_metrics.ERROR_MSG] = error_msg
                metrics[common_metrics.ERROR_CODE] = error_response_code
                print(f"Warning Or Error: {e}")
                print("error_response_code",error_response_code)

            metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token) #This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
            metrics[common_metrics.TTFT] = ttft
            metrics[common_metrics.E2E_LAT] = total_request_time
            metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
            metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
            metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
            metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len
            metrics[common_metrics.ENDPOINT]=address
            metrics[common_metrics.PROVIDER_PLATFORM]=azure
            metrics[common_metrics.REQUEST]=body_content
            metrics[common_metrics.RESPONSE]=body_response
            metrics[common_metrics.REQUEST_ID]=request_id

            return metrics, generated_text, request_config
