from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential


class IdentityService:
    
    @staticmethod
    def get_azure_credentials(secret_key: str) -> AzureKeyCredential:
        return AzureKeyCredential(secret_key)
