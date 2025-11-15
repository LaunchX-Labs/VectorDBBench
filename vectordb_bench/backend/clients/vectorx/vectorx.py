import logging
from contextlib import contextmanager

from vecx import vectorx

from ..api import DBCaseConfig, EmptyDBCaseConfig, IndexType, VectorDB
from .config import VectorXConfig

log = logging.getLogger(__name__)


class VectorX(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        print(db_config)

        self.token = db_config.get("token", "")
        self.region = db_config.get("region", "india-west-1")
        
        self.collection_name = db_config.get("collection_name", "")
        if not self.collection_name:
            import uuid
            self.collection_name = f"vectorx_bench_{uuid.uuid4().hex[:8]}"
        
        self.key = db_config.get("key")
        self.use_encryption = db_config.get("use_encryption")
        self.space_type = db_config.get("space_type", "cosine")
        self.use_fp16 = db_config.get("use_fp16")
        self.version = db_config.get("version")
        self.M = db_config.get("m")
        self.ef_con = db_config.get("ef_con")
        self.ef_search = db_config.get("ef_search")
        self.vx = vectorx.VectorX(token=self.token)
        
        try:
            indices = self.vx.list_indexes().get("indices", [])
            index_names = [index["name"] for index in indices] if indices else []
            
            if drop_old and self.collection_name in index_names:
                self._drop_index(self.collection_name)
                self._create_index(dim)
            elif self.collection_name not in index_names:
                self._create_index(dim)
        except Exception as e:
            log.error(f"Error connecting to VectorX API: {e}")
            raise

    def _create_index(self, dim):
        try:
            resp = self.vx.create_index(
                name=self.collection_name, 
                key=self.key if self.use_encryption else None, 
                dimension=dim,
                space_type=self.space_type,
                use_fp16=self.use_fp16,
                version=self.version,
                M=self.M,
                ef_con=self.ef_con
            )
            log.info(f"Created new VectorX index: {resp}")
        except Exception as e:
            log.error(f"Failed to create VectorX index: {e}")
            raise

    def _drop_index(self, collection_name):
        try:
            res = self.vx.delete_index(collection_name)
            log.info(res)
        except Exception as e:
            log.error(f"Failed to drop VectorX index: {e}")
            raise

    @classmethod
    def config_cls(cls) -> type[VectorXConfig]:
        return VectorXConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> type[DBCaseConfig]:
        return EmptyDBCaseConfig

    @contextmanager
    def init(self):
        try:
            # log.info(f"Token: {self.token}")
            vx = vectorx.VectorX(token=self.token)
            self.index = vx.get_index(name=self.collection_name, key=self.key if self.use_encryption else None)
            yield
        except Exception as e:
            log.error(f"Error initializing VectorX client: {e}")
            raise
        finally:
            pass

    def optimize(self, data_size: int | None = None):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception): # type: ignore
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            batch_vectors = [
                {
                    "id": str(metadata[i]),
                    "vector": embeddings[i],
                    "meta": {"id": str(metadata[i])}
                }
                for i in range(len(embeddings))
            ]
            self.index.upsert(batch_vectors)
            insert_count = len(batch_vectors)
            
        except Exception as e:
            return (insert_count, e)
            
        return (len(embeddings), None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        **kwargs,
    ) -> list[int]:
        try:
            filter_expr = None
            if filters and "id" in filters:
                filter_expr = {"id": filters["id"]}
                
            results = self.index.query(
                vector=query,
                top_k=k,
                filter=filter_expr,
                ef=self.ef_search,
                include_vectors=False
            )
            
            return [int(result["id"]) for result in results]
            
        except Exception as e:
            log.warning(f"Error querying VectorX index: {e}")
            raise

    def describe_index(self) -> dict:
        """Get information about the current index."""
        try:
            all_indices = self.vx.list_indexes().get("indices", [])
            
            for idx in all_indices:
                if idx.get("name") == self.collection_name:
                    return idx
                    
            return {}
        except Exception as e:
            log.warning(f"Error describing VectorX index: {e}")
            return {}
