"""
QData Transformer - 高性能、可扩展的数据转换引擎

基于 Polars 和 DuckDB，提供卓越的数据处理性能。

基本使用：
    >>> import polars as pl
    >>> from qdata_transformer import PolarsFieldMappingTransformer
    >>>
    >>> data = pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    >>> transformer = PolarsFieldMappingTransformer()
    >>> result = transformer.execute(data, {
    ...     "mappings": [{"source": "name", "target": "user_name"}]
    ... })

转换链使用：
    >>> from qdata_transformer import TransformChain
    >>>
    >>> chain = (
    ...     TransformChain()
    ...     .add("polars_field_mapping", {"mappings": [...]})
    ...     .add("duckdb_aggregation", {"group_by": [...], "aggregations": [...]})
    ... )
    >>> result = chain.execute(data)
"""

__version__ = "1.0.0"
__author__ = "广东轻亿云软件科技有限公司"
__email__ = "opensource@qeasy.cloud"

# 核心类
from qdata_transformer.base import (
    BaseTransformer,
    TransformChain,
    TransformResult,
    TransformStep,
)

# 注册中心
from qdata_transformer.registry import TransformerRegistry

# 异常类
from qdata_transformer.exceptions import (
    AggregationConfigError,
    InvalidColumnError,
    MappingConfigError,
    TransformerConfigError,
    TransformerError,
    TransformerNotFoundError,
    TransformExecutionError,
)

# 转换器
from qdata_transformer.transformers.mapping import PolarsFieldMappingTransformer
from qdata_transformer.transformers.multi_mapping import PolarsMultiMappingTransformer
from qdata_transformer.transformers.aggregation import (
    DuckDBAggregationTransformer,
    DuckDBSQLTransformer,
)

# 公开的 API
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    # 核心类
    "BaseTransformer",
    "TransformChain",
    "TransformResult",
    "TransformStep",
    # 注册中心
    "TransformerRegistry",
    # 异常类
    "TransformerError",
    "TransformerNotFoundError",
    "TransformerConfigError",
    "TransformExecutionError",
    "MappingConfigError",
    "AggregationConfigError",
    "InvalidColumnError",
    # 转换器
    "PolarsFieldMappingTransformer",
    "PolarsMultiMappingTransformer",
    "DuckDBAggregationTransformer",
    "DuckDBSQLTransformer",
]
