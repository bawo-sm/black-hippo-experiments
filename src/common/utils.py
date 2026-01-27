import os
from typing import Any
from azure.storage.blob import BlobServiceClient
# from azure.servicebus ...
# from azure.eventgrid ...
from azure.identity import DefaultAzureCredential
from src.common.enums import AzureClientEnum


def get_azure_client(service_name: AzureClientEnum) -> BlobServiceClient | DefaultAzureCredential:
    """
    Creates Azure client to the given cloud service. 
    Handles common errors.
    """

    match service_name:
        case AzureClientEnum.blob:
            pass

def get_env_variable(name: str) -> Any:
    assert isinstance(name, str)
    try:
        return os.environ[name]
    except KeyError:
        listed_names = "   ".join([x for x in os.environ.keys()])
        raise KeyError(
            f"\nThere is no variable {name} in the current environment.\nCurrent variables are:\n{listed_names}"
        )
