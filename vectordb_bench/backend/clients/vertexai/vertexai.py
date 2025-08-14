"""Wrapper around the Vertex AI Vector Search over VectorDB"""

import logging
import os
from contextlib import contextmanager
from collections.abc import Iterable
from typing import List, Optional

from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud.aiplatform_v1.types import UpsertDatapointsRequest, IndexDatapoint
from google.api_core.exceptions import GoogleAPIError

from vectordb_bench.backend.filter import Filter, FilterOp
from ..api import VectorDB
from .config import VertexAIConfig

log = logging.getLogger(__name__)

VERTEXAI_MAX_NUM_PER_BATCH = 1000  # Recommended max batch size


class VertexAIVectorDB(VectorDB):
    """Vertex AI Vector Search implementation for VectorDBBench."""

    supported_filter_types: List[FilterOp] = [
        FilterOp.NonFilter,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: VertexAIConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        # Store config values only (no client init here)
        self.db_config = db_config
        self.case_config = db_case_config
        self.with_scalar_labels = with_scalar_labels
        self.batch_size = VERTEXAI_MAX_NUM_PER_BATCH
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self.project_id = db_config.get("project_id")
        self.region = db_config.get("region")
        self.index_id = db_config.get("index_id")
        self.endpoint_id = db_config.get("endpoint_id")
        self.dimension = dim
        self.gcp_service_account = db_config.get("gcp_service_account")

        if self.gcp_service_account:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.gcp_service_account

        # Lazy client placeholders
        self.index = None
        self.endpoint = None
        self.index_client = None
        self.match_client = None
        self.deployed_index_id = None



        # Auto-create if index/endpoint not provided
        if not self.index_id or not self.endpoint_id:
            self._ensure_index_and_endpoint()

    def _ensure_index_and_endpoint(self):
        """Create streaming-enabled index/endpoint if missing."""
        aiplatform.init(project=self.project_id, location=self.region)

        if not self.index_id:
            log.info("Creating new streaming-enabled index...")
            idx = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=f"auto-index-{self.dimension}d",
                dimensions=self.dimension,
                # distance_measure_type="COSINE_DISTANCE",
                distance_measure_type="DOT_PRODUCT_DISTANCE",
                shard_size="SHARD_SIZE_SMALL",
                description="Auto-created index with streaming update enabled",
                approximate_neighbors_count=100,  # <--- Add this required parameter
                index_update_method="STREAM_UPDATE",  # <-- Official way to enable streaming
            )
            idx.wait()
            self.index = idx   # Assign here!
            self.index_id = idx.resource_name.split("/")[-1]
            log.info(f"Created index: {self.index_id}")

        if not self.endpoint_id:
            log.info("Creating new index endpoint...")
            ep = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name="auto-endpoint",
                description="Auto-created endpoint for benchmarking",
                public_endpoint_enabled=True
            )
            ep.wait()
            self.endpoint_id = ep.resource_name.split("/")[-1]
            log.info(f"Created endpoint: {self.endpoint_id}")

            log.info("Deploying index to endpoint...")
            ep.deploy_index(index=self.index, deployed_index_id="deployed-index-1").result()
            log.info("Index deployed.")





    def _init_clients(self):
        """Initialize Vertex AI clients lazily in worker process."""
        if self.index is None or self.endpoint is None:
            aiplatform.init(project=self.project_id, location=self.region)
            self.index = aiplatform.MatchingEngineIndex(index_name=self.index_id)
            self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self.endpoint_id
            )
            if not self.endpoint.deployed_indexes:
                raise RuntimeError("No deployed indexes found on the endpoint.")
            self.deployed_index_id = self.endpoint.deployed_indexes[0].id

        if self.index_client is None:
            self.index_client = aiplatform_v1.IndexServiceClient(
                client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com:443"}
            )

        # Ensure MatchService client is configured to the endpoint's public domain when available
        if self.match_client is None:
            public_domain = None
            # Wrapper usually exposes property; fall back to gca_resource if needed
            try:
                public_domain = getattr(self.endpoint, "public_endpoint_domain_name", None)
            except Exception:
                public_domain = None
            if not public_domain:
                try:
                    public_domain = getattr(self.endpoint, "gca_resource", None)
                    if public_domain is not None:
                        public_domain = getattr(self.endpoint.gca_resource, "public_endpoint_domain_name", None)
                except Exception:
                    public_domain = None

            api_host = (
                f"{public_domain}:443" if public_domain else f"{self.region}-aiplatform.googleapis.com:443"
            )
            self.match_client = aiplatform_v1.MatchServiceClient(
                client_options={"api_endpoint": api_host}
            )

    @contextmanager
    def init(self):
        yield

    def optimize(self, **kwargs):
        return

    def need_normalize_cosine(self) -> bool:
        return False

    # def delete_embeddings(self, ids: List[int]):
    #     """Delete datapoints from the Vertex AI index by their IDs."""
    #     self._init_clients()
    #     datapoint_ids = [str(i) for i in ids]
    #     index_name = self.index_client.index_path(self.project_id, self.region, self.index_id)
        
    #     try:
    #         self.index_client.remove_datapoints(index=index_name, datapoint_ids=datapoint_ids)
    #         log.info(f"Deleted {len(datapoint_ids)} datapoints from index.")
    #     except Exception as e:
    #         log.error(f"Failed to delete datapoints: {e}")
    #         raise


    def insert_embeddings(
        self,
        embeddings: Iterable[List[float]],
        metadata: List[int],
        labels_data: Optional[List[str]] = None,
        **kwargs,
    ) -> tuple[int, Optional[Exception]]:
        """Batch upsert embeddings into Vertex AI Vector Search."""
        self._init_clients()

        assert len(embeddings) == len(metadata)
        if self.with_scalar_labels:
            assert labels_data is not None and len(labels_data) == len(metadata)

        insert_count = 0
        vectors = list(embeddings)
        datapoint_ids = [str(id_) for id_ in metadata]

        index_name = self.index_client.index_path(self.project_id, self.region, self.index_id)

        try:
            for batch_start in range(0, len(vectors), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(vectors))
                batch_vectors = vectors[batch_start:batch_end]
                batch_ids = datapoint_ids[batch_start:batch_end]
                batch_labels = (
                    labels_data[batch_start:batch_end]
                    if self.with_scalar_labels else None
                )

                datapoints = []
                for i, emb in enumerate(batch_vectors):
                    # Build IndexDatapoint with dicts for feature_vector and restricts
                    dp = IndexDatapoint(
                        datapoint_id=batch_ids[i],
                        # feature_vector={"values": emb},
                        feature_vector=emb,
                        restricts=[
                            {"namespace": self._scalar_id_field, "allow_list": [batch_ids[i]]},
                            *(
                                [{"namespace": self._scalar_label_field, "allow_list": [batch_labels[i]]}]
                                if self.with_scalar_labels and batch_labels else []
                            )
                        ],
                    )
                    datapoints.append(dp)

                request = UpsertDatapointsRequest(index=index_name, datapoints=datapoints)
                try:
                    self.index_client.upsert_datapoints(request=request)
                    insert_count += len(datapoints)
                except GoogleAPIError as e:
                    log.error(f"Batch upsert failed ({batch_start}-{batch_end}): {e}")
                    return insert_count, e

        except Exception as e:
            log.error(f"Failed to insert embeddings: {e}")
            return insert_count, e

        return insert_count, None

    def search_embedding(
        self,
        query: List[float],
        k: int = 100,
        timeout: Optional[int] = None,
    ) -> List[int]:
        """Perform ANN search for a single embedding."""
        self._init_clients()
        try:
            # Prefer direct MatchService call bound to the endpoint's public domain
            request = {
                "index_endpoint": self.endpoint.resource_name,
                "deployed_index_id": self.deployed_index_id,
                "queries": [
                    {
                        "datapoint": {"feature_vector": query},
                        "neighbor_count": k,
                    }
                ],
            }
            response = self.match_client.find_neighbors(request=request, timeout=timeout)

            # Parse response for both proto and dict styles
            try:
                neighbors = response.nearest_neighbors[0].neighbors
                return [int(n.datapoint.datapoint_id) for n in neighbors]
            except Exception:
                raw_neighbors = response["nearest_neighbors"][0]["neighbors"]
                return [int(n["datapoint"]["datapoint_id"]) for n in raw_neighbors]
        except Exception as e:
            log.error(
                "Search failed via MatchService. If your endpoint is private, enable public_endpoint or configure PSC. Error: %s",
                e,
            )
            return []

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = None
        else:
            raise ValueError(f"Filter type not supported for Vertex AI: {filters}")

