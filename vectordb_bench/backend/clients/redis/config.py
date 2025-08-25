from pydantic import BaseModel, SecretStr, validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class RedisConfig(DBConfig):
    password: SecretStr | None = None
    host: SecretStr
    port: int | None = None

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "password": self.password.get_secret_value() if self.password is not None else None,
        }


class RedisIndexConfig(BaseModel):
    """Base config for milvus"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""
        return self.metric_type.value


class RedisHNSWConfig(RedisIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    # M: int = 32
    # efConstruction: int = 256
    # ef: int | None = 128
    index: IndexType = IndexType.HNSW

    @validator('M', 'efConstruction')
    def must_be_positive(cls, v):
        if v is None or v <= 0:
            raise ValueError('M and efConstruction must be positive integers')
        return v

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        if self.ef is None:
            return {"metric_type": self.parse_metric(), "params": {}}
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef},
        }
