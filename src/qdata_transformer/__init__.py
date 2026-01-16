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

可用转换器：
    Polars 转换器（高性能向量化处理）：
    - polars_field_mapping: 字段映射转换
    - polars_multi_mapping: 批量映射转换
    - polars_filter: 数据过滤
    - polars_column_ops: 列运算
    - polars_split: 数据拆分
    - polars_deduplicate: 数据去重
    - polars_merge: 数据合并

    DuckDB 转换器（SQL 分析能力）：
    - duckdb_aggregation: SQL 聚合
    - duckdb_sql: 自定义 SQL
    - duckdb_join: SQL JOIN
    - duckdb_window: 窗口函数
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

# Polars 转换器
from qdata_transformer.transformers.mapping import PolarsFieldMappingTransformer
from qdata_transformer.transformers.multi_mapping import PolarsMultiMappingTransformer
from qdata_transformer.transformers.filter import PolarsFilterTransformer
from qdata_transformer.transformers.column_ops import PolarsColumnOpsTransformer
from qdata_transformer.transformers.split import (
    PolarsSplitTransformer,
    PolarsDeduplicateTransformer,
    PolarsMergeTransformer,
)

# DuckDB 转换器
from qdata_transformer.transformers.aggregation import (
    DuckDBAggregationTransformer,
    DuckDBSQLTransformer,
)
from qdata_transformer.transformers.advanced import (
    DuckDBJoinTransformer,
    DuckDBWindowTransformer,
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
    # Polars 转换器
    "PolarsFieldMappingTransformer",
    "PolarsMultiMappingTransformer",
    "PolarsFilterTransformer",
    "PolarsColumnOpsTransformer",
    "PolarsSplitTransformer",
    "PolarsDeduplicateTransformer",
    "PolarsMergeTransformer",
    # DuckDB 转换器
    "DuckDBAggregationTransformer",
    "DuckDBSQLTransformer",
    "DuckDBJoinTransformer",
    "DuckDBWindowTransformer",
]
