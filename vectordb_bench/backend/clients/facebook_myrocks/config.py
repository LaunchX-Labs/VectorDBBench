# from abc import abstractmethod
# from collections.abc import Mapping, Sequence
# from typing import Any, LiteralString, TypedDict

# from pydantic import BaseModel, SecretStr

# from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

# MYSQL_URL_PLACEHOLDER = "mysql://%s:%s@%s:%d/%s"


# class FacebookMyRocksConfigDict(TypedDict):
#     """These keys will be directly used as kwargs in mysql.connector connection,
#     so the names must match exactly mysql.connector API"""

#     user: str
#     password: str
#     host: str
#     port: int
#     database: str


# class FacebookMyRocksConfig(DBConfig):
#     user_name: SecretStr = SecretStr("root")
#     password: SecretStr
#     host: str = "localhost"
#     port: int = 3306
#     db_name: str

#     def to_dict(self) -> FacebookMyRocksConfigDict:
#         user_str = self.user_name.get_secret_value()
#         pwd_str = self.password.get_secret_value()
#         return {
#             "host": self.host,
#             "port": self.port,
#             "database": self.db_name,
#             "user": user_str,
#             "password": pwd_str,
#         }


# class FacebookMyRocksIndexParam(TypedDict):
#     vector_type: str  # "BLOB" or "JSON"
#     index_type: str  # "flat", "ivfflat", "ivfpq"
#     trained_index_table: str | None  # For IVF types
#     trained_index_id: int | None  # For IVF types
#     metric_type: str  # "L2", "IP", "COSINE"


# class FacebookMyRocksSearchParam(TypedDict):
#     metric_type: str  # "L2", "IP", "COSINE"


# class FacebookMyRocksIndexConfig(BaseModel, DBCaseConfig):
#     metric_type: MetricType | None = None
#     vector_type: str = "BLOB"  # "BLOB" (recommended) or "JSON"
#     create_index_before_load: bool = False
#     create_index_after_load: bool = True

#     def parse_metric_type(self) -> str:
#         """Convert MetricType enum to Facebook MyRocks metric string"""
#         metric_mapping = {
#             MetricType.L2: "L2",
#             MetricType.IP: "IP", 
#             MetricType.COSINE: "COSINE",
#         }
#         return metric_mapping.get(self.metric_type, "L2")

#     @abstractmethod
#     def index_param(self) -> FacebookMyRocksIndexParam: ...

#     @abstractmethod 
#     def search_param(self) -> FacebookMyRocksSearchParam: ...


# class FacebookMyRocksFlatConfig(FacebookMyRocksIndexConfig):
#     """
#     Flat index configuration for Facebook MyRocks.
#     This is the simplest index type that performs exact search.
#     Good for smaller datasets or when exact results are required.
#     """
    
#     index: IndexType = IndexType.FLAT

#     def index_param(self) -> FacebookMyRocksIndexParam:
#         return {
#             "vector_type": self.vector_type,
#             "index_type": "flat",
#             "trained_index_table": None,
#             "trained_index_id": None,
#             "metric_type": self.parse_metric_type(),
#         }

#     def search_param(self) -> FacebookMyRocksSearchParam:
#         return {
#             "metric_type": self.parse_metric_type(),
#         }


# class FacebookMyRocksIVFFlatConfig(FacebookMyRocksIndexConfig):
#     """
#     IVFFlat index configuration for Facebook MyRocks.
#     Uses Inverted File (IVF) with flat quantization.
#     Requires training on sample data before use.
#     """
    
#     # IVF parameters
#     nlist: int | None = None  # Number of clusters/centroids
#     nprobe: int | None = None  # Number of clusters to search
#     trained_index_table: str | None = None  # Table to store trained index
#     trained_index_id: int = 1  # ID for the trained index
    
#     index: IndexType = IndexType.IVFFlat

#     def index_param(self) -> FacebookMyRocksIndexParam:
#         return {
#             "vector_type": self.vector_type,
#             "index_type": "ivfflat", 
#             "trained_index_table": self.trained_index_table,
#             "trained_index_id": self.trained_index_id,
#             "metric_type": self.parse_metric_type(),
#             "nlist": self.nlist,
#         }

#     def search_param(self) -> FacebookMyRocksSearchParam:
#         return {
#             "metric_type": self.parse_metric_type(),
#             "nprobe": self.nprobe,
#         }


# class FacebookMyRocksIVFPQConfig(FacebookMyRocksIndexConfig):
#     """
#     IVFPQ index configuration for Facebook MyRocks.
#     Uses Inverted File (IVF) with Product Quantization (PQ).
#     Most memory efficient but with some accuracy trade-off.
#     Requires training on sample data before use.
#     """
    
#     # IVF parameters
#     nlist: int | None = None  # Number of clusters/centroids
#     nprobe: int | None = None  # Number of clusters to search
    
#     # PQ parameters  
#     m: int | None = None  # Number of subquantizers
#     nbits: int | None = None  # Number of bits per subquantizer
    
#     trained_index_table: str | None = None  # Table to store trained index
#     trained_index_id: int = 1  # ID for the trained index
    
#     index: IndexType = IndexType.IVFPQ

#     def index_param(self) -> FacebookMyRocksIndexParam:
#         return {
#             "vector_type": self.vector_type,
#             "index_type": "ivfpq",
#             "trained_index_table": self.trained_index_table, 
#             "trained_index_id": self.trained_index_id,
#             "metric_type": self.parse_metric_type(),
#             "nlist": self.nlist,
#             "m": self.m,
#             "nbits": self.nbits,
#         }

#     def search_param(self) -> FacebookMyRocksSearchParam:
#         return {
#             "metric_type": self.parse_metric_type(),
#             "nprobe": self.nprobe,
#         }


# # Mapping of index types to config classes
# _facebook_myrocks_case_config = {
#     IndexType.FLAT: FacebookMyRocksFlatConfig,
#     IndexType.IVFFlat: FacebookMyRocksIVFFlatConfig, 
#     IndexType.IVFPQ: FacebookMyRocksIVFPQConfig,
# }

#-------------------------------------------------------------------------------
# from abc import abstractmethod
# from collections.abc import Mapping, Sequence
# from typing import Any, LiteralString, TypedDict

# from pydantic import BaseModel, SecretStr

# from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

# MYSQL_URL_PLACEHOLDER = "mysql://%s:%s@%s:%d/%s"


# class FacebookMyRocksConfigDict(TypedDict):
#     """These keys will be directly used as kwargs in mysql.connector connection,
#     so the names must match exactly mysql.connector API"""

#     user: str
#     password: str
#     host: str
#     port: int
#     database: str


# class FacebookMyRocksConfig(DBConfig):
#     user_name: SecretStr = SecretStr("root")
#     password: SecretStr
#     host: str = "localhost"
#     port: int = 3306
#     db_name: str

#     def to_dict(self) -> FacebookMyRocksConfigDict:
#         user_str = self.user_name.get_secret_value()
#         pwd_str = self.password.get_secret_value()
#         return {
#             "host": self.host,
#             "port": self.port,
#             "database": self.db_name,
#             "user": user_str,
#             "password": pwd_str,
#         }


# class FacebookMyRocksIndexParam(TypedDict):
#     vector_type: str  # "BLOB" or "JSON"
#     index_type: str  # "flat", "ivfflat", "ivfpq"
#     trained_index_table: str | None  # For IVF types
#     trained_index_id: int | None  # For IVF types
#     metric_type: str  # "L2", "IP", "COSINE"


# class FacebookMyRocksSearchParam(TypedDict):
#     metric_type: str  # "L2", "IP", "COSINE"


# class FacebookMyRocksIndexConfig(BaseModel, DBCaseConfig):
#     metric_type: MetricType | None = None
#     vector_type: str = "BLOB"  # "BLOB" (recommended) or "JSON"
#     create_index_before_load: bool = False
#     create_index_after_load: bool = True
#     train_index_after_load: bool = True  # Train IVF indexes after data loading

#     def parse_metric_type(self) -> str:
#         """Convert MetricType enum to Facebook MyRocks metric string"""
#         metric_mapping = {
#             MetricType.L2: "L2",
#             MetricType.IP: "IP", 
#             MetricType.COSINE: "COSINE",
#         }
#         return metric_mapping.get(self.metric_type, "L2")

#     # @abstractmethod
#     # def index_param(self) -> FacebookMyRocksIndexParam: ...

#     # @abstractmethod 
#     # def search_param(self) -> FacebookMyRocksSearchParam: ...
#     # def index_param(self) -> FacebookMyRocksIndexParam:
#     #     return {
#     #         "vector_type": self.vector_type,
#     #         "index_type": self.index_type,
#     #         "trained_index_table": self.trained_index_table,
#     #         "trained_index_id": self.trained_index_id,
#     #         "metric_type": self.parse_metric_type(),
#     #     }
#     def index_param(self) -> FacebookMyRocksIndexParam:
#         raise NotImplementedError("index_param must be implemented by subclass.")

#     def search_param(self) -> FacebookMyRocksSearchParam:
#         return {
#             "metric_type": self.parse_metric_type(),
#         }



# class FacebookMyRocksFlatConfig(FacebookMyRocksIndexConfig):
#     """
#     Flat index configuration for Facebook MyRocks.
#     This is the simplest index type that performs exact search.
#     Good for smaller datasets or when exact results are required.
#     """
    
#     index: IndexType = IndexType.Flat

#     def index_param(self) -> FacebookMyRocksIndexParam:
#         return {
#             "vector_type": self.vector_type,
#             "index_type": "flat",
#             "trained_index_table": None,
#             "trained_index_id": None,
#             "metric_type": self.parse_metric_type(),
#         }

#     def search_param(self) -> FacebookMyRocksSearchParam:
#         return {
#             "metric_type": self.parse_metric_type(),
#         }


# class FacebookMyRocksIVFFlatConfig(FacebookMyRocksIndexConfig):
#     """
#     IVFFlat index configuration for Facebook MyRocks.
#     Uses Inverted File (IVF) with flat quantization.
#     Requires training on sample data before use.
#     """
    
#     # IVF parameters
#     nlist: int | None = None  # Number of clusters/centroids
#     nprobe: int | None = None  # Number of clusters to search
#     trained_index_table: str | None = None  # Table to store trained index
#     trained_index_id: int = 1  # ID for the trained index
    
#     index: IndexType = IndexType.IVFFlat

#     def index_param(self) -> FacebookMyRocksIndexParam:
#         return {
#             "vector_type": self.vector_type,
#             "index_type": "ivfflat", 
#             "trained_index_table": self.trained_index_table,
#             "trained_index_id": self.trained_index_id,
#             "metric_type": self.parse_metric_type(),
#             "nlist": self.nlist,
#         }

#     def search_param(self) -> FacebookMyRocksSearchParam:
#         return {
#             "metric_type": self.parse_metric_type(),
#             "nprobe": self.nprobe,
#         }


# class FacebookMyRocksIVFPQConfig(FacebookMyRocksIndexConfig):
#     """
#     IVFPQ index configuration for Facebook MyRocks.
#     Uses Inverted File (IVF) with Product Quantization (PQ).
#     Most memory efficient but with some accuracy trade-off.
#     Requires training on sample data before use.
#     """
    
#     # IVF parameters
#     nlist: int | None = None  # Number of clusters/centroids
#     nprobe: int | None = None  # Number of clusters to search
    
#     # PQ parameters  
#     m: int | None = None  # Number of subquantizers
#     nbits: int | None = None  # Number of bits per subquantizer
    
#     trained_index_table: str | None = None  # Table to store trained index
#     trained_index_id: int = 1  # ID for the trained index
    
#     index: IndexType = IndexType.IVFPQ

#     def index_param(self) -> FacebookMyRocksIndexParam:
#         return {
#             "vector_type": self.vector_type,
#             "index_type": "ivfpq",
#             "trained_index_table": self.trained_index_table, 
#             "trained_index_id": self.trained_index_id,
#             "metric_type": self.parse_metric_type(),
#             "nlist": self.nlist,
#             "m": self.m,
#             "nbits": self.nbits,
#         }

#     def search_param(self) -> FacebookMyRocksSearchParam:
#         return {
#             "metric_type": self.parse_metric_type(),
#             "nprobe": self.nprobe,
#         }


# # Mapping of index types to config classes
# _facebook_myrocks_case_config = {
#     IndexType.Flat: FacebookMyRocksFlatConfig,
#     IndexType.IVFFlat: FacebookMyRocksIVFFlatConfig, 
#     IndexType.IVFPQ: FacebookMyRocksIVFPQConfig,
# }

#-----------------------------------------------------------
from pydantic import SecretStr
from typing import Optional
from ..api import DBConfig


class FacebookMyRocksConfig(DBConfig):
    user: str = "root"
    password: SecretStr
    host: str = "localhost"
    port: int = 3306
    database: str = "rocksdb"
    vector_type: str = "BLOB" #or "JSON"
    index_type: Optional[str] = "flat" 
    metric_type: str = "COSINE"
    dimension: int = 768

    def to_dict(self) -> dict:
        return {
            "user": self.user,
            "password": self.password.get_secret_value(),  # Decrypt SecretStr
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "vector_type": self.vector_type,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "dimension": self.dimension
        }


