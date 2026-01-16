"""
转换器子包

包含各种具体的转换器实现。

提供以下类型的转换器：
- 字段映射：PolarsFieldMappingTransformer, PolarsMultiMappingTransformer
- 数据过滤：PolarsFilterTransformer
- 列运算：PolarsColumnOpsTransformer
- 数据拆分：PolarsSplitTransformer
- 数据去重：PolarsDeduplicateTransformer
- 数据合并：PolarsMergeTransformer
- SQL 聚合：DuckDBAggregationTransformer, DuckDBSQLTransformer
- SQL JOIN：DuckDBJoinTransformer
- 窗口函数：DuckDBWindowTransformer
"""

from qdata_transformer.transformers.mapping import PolarsFieldMappingTransformer
from qdata_transformer.transformers.multi_mapping import PolarsMultiMappingTransformer
from qdata_transformer.transformers.aggregation import (
    DuckDBAggregationTransformer,
    DuckDBSQLTransformer,
)
from qdata_transformer.transformers.filter import PolarsFilterTransformer
from qdata_transformer.transformers.column_ops import PolarsColumnOpsTransformer
from qdata_transformer.transformers.split import (
    PolarsSplitTransformer,
    PolarsDeduplicateTransformer,
    PolarsMergeTransformer,
)
from qdata_transformer.transformers.advanced import (
    DuckDBJoinTransformer,
    DuckDBWindowTransformer,
)

__all__ = [
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
