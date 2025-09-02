from pydantic import SecretStr
from typing import Optional
from ..api import DBConfig


class VectorXConfig(DBConfig):
    token: SecretStr
    region: Optional[str] = "india-west-1"
    key: Optional[str] = "3a5f08c7d9e1b2a43a5f08c7d9e1b2a4"
    space_type: str
    use_fp16: bool = True
    use_encryption: bool = True
    version: Optional[int] = 1
    m: Optional[int] = 32
    ef_con: Optional[int] = 256
    ef_search: Optional[int] = 128
    collection_name: str
    
    def to_dict(self) -> dict:
        return {
            "token": self.token.get_secret_value(),
            "region": self.region,
            "key": self.key,
            "space_type": self.space_type,
            "use_fp16": self.use_fp16,
            "use_encryption": self.use_encryption,
            "version": self.version,
            "m": self.m,
            "ef_con": self.ef_con,
            "ef_search": self.ef_search,
            "collection_name": self.collection_name,
        }