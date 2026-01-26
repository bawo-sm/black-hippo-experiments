from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential


class IdentityService:
    
    @staticmethod
    def get_azure_crdentials(secret_key: str) -> AzureKeyCredential:
        return AzureKeyCredential(secret_key)
