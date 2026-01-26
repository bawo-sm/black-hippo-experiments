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
