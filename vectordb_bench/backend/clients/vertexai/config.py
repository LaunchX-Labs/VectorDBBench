from pydantic import SecretStr, BaseModel
from typing import Optional

from ..api import DBConfig, DBCaseConfig, MetricType

class VertexAIConfig(DBConfig):
    """Vertex AI project-level configuration (for client creation)."""
    project_id: str
    region: str = "us-central1"
    # index_id: str
    # endpoint_id: str
    index_id: Optional[str] = None    # optional now
    endpoint_id: Optional[str] = None # optional now
    gcp_service_account: SecretStr | None = None  # optional service account json path

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "region": self.region,
            # "index_id": self.index_id,
            # "endpoint_id": self.endpoint_id,
            "index_id": self.index_id or "",
            "endpoint_id": self.endpoint_id or "",
            "gcp_service_account": (
                self.gcp_service_account.get_secret_value() if self.gcp_service_account else ""
            ),
        }

class VertexAIIndexConfig(DBCaseConfig, BaseModel):
    """
    Per-benchmark-case parameters for Vertex AI Vector Search.
    """
    # metric_type: MetricType | None = None
    # metric_type: MetricType = MetricType.COSINE
    metric_type: MetricType = MetricType.IP  # Default to Inner Product (IP), can be overridden

    # dimension: int
    dimension: int = 768  # Default dimension, can be overridden

    # Optionally, map more params if needed (Vertex AI hides many ANN params, but you might expose num_shards/gcs_bucket or similar).
    gcs_bucket: str | None = None

    def parse_metric(self) -> str:
        # Vertex AI supports:
        # - "squared_l2_distance"
        # - "dot_product_distance"
        # - "cosine_distance"
        if self.metric_type == MetricType.L2:
            return "squared_l2_distance"
        if self.metric_type == MetricType.IP:
            return "dot_product_distance"
        # return "cosine_distance"
        return "dot_product_distance"

    def index_param(self) -> dict:
        out = {
            "dimension": self.dimension,
            "distance_measure_type": self.parse_metric(),
        }
        # Add any additional index init params here if necessary
        if self.gcs_bucket:
            out["gcs_bucket"] = self.gcs_bucket
        return out

    def search_param(self) -> dict:
        return {
            # You could add options like search_algorithm="auto", num_neighbors=10, etc. as Vertex AI compatibility grows.
        }
