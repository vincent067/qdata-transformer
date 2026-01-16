"""
DuckDB 聚合转换器

实现基于 DuckDB 的 SQL 聚合转换，支持：
- 分组聚合：GROUP BY + 聚合函数
- 窗口函数：ROW_NUMBER, RANK 等
- 自定义 SQL 查询
- HAVING 过滤
"""

from typing import Any, ClassVar, Dict, List, Optional, Set

import duckdb
import polars as pl

from qdata_transformer.base import BaseTransformer
from qdata_transformer.exceptions import AggregationConfigError, TransformExecutionError
from qdata_transformer.registry import TransformerRegistry


@TransformerRegistry.register("duckdb_aggregation")
class DuckDBAggregationTransformer(BaseTransformer):
    """
    DuckDB 聚合转换器

    提供高性能的 SQL 聚合能力。

    配置示例：
        {
            "group_by": ["customer_id", "order_date"],
            "aggregations": [
                {"field": "amount", "function": "sum", "alias": "total_amount"},
                {"field": "id", "function": "count", "alias": "order_count"},
                {"field": "amount", "function": "avg", "alias": "avg_amount"}
            ],
            "having": "total_amount > 1000",
            "order_by": ["total_amount desc"]
        }

    支持的聚合函数：
        - count: 计数
        - sum: 求和
        - avg: 平均值
        - min: 最小值
        - max: 最大值
        - first: 第一个值
        - last: 最后一个值
        - count_distinct: 去重计数
        - string_agg: 字符串聚合
        - list: 列表聚合
        - median: 中位数
        - stddev: 标准差
        - variance: 方差

    使用示例：
        transformer = DuckDBAggregationTransformer()
        result = transformer.execute(df, {
            "group_by": ["customer_id"],
            "aggregations": [
                {"field": "amount", "function": "sum", "alias": "total"}
            ]
        })
    """

    name: ClassVar[str] = "duckdb_aggregation"
    description: ClassVar[str] = "DuckDB SQL 聚合转换"
    version: ClassVar[str] = "1.0.0"

    # 支持的聚合函数
    SUPPORTED_FUNCTIONS: Set[str] = {
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "first",
        "last",
        "count_distinct",
        "string_agg",
        "list",
        "median",
        "stddev",
        "variance",
    }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证聚合配置"""
        # group_by 可选，如果没有则是全局聚合
        group_by = config.get("group_by", [])
        if not isinstance(group_by, list):
            raise AggregationConfigError("'group_by' 必须是列表")

        # aggregations 必须有
        if "aggregations" not in config:
            raise AggregationConfigError("配置缺少 'aggregations' 字段")

        aggregations = config["aggregations"]
        if not isinstance(aggregations, list):
            raise AggregationConfigError("'aggregations' 必须是列表")

        if len(aggregations) == 0:
            raise AggregationConfigError("'aggregations' 不能为空")

        for i, agg in enumerate(aggregations):
            if not isinstance(agg, dict):
                raise AggregationConfigError(f"第 {i + 1} 个聚合配置必须是字典")

            if "field" not in agg:
                raise AggregationConfigError(f"第 {i + 1} 个聚合配置缺少 'field' 字段")

            if "function" not in agg:
                raise AggregationConfigError(f"第 {i + 1} 个聚合配置缺少 'function' 字段")

            func = agg["function"].lower()
            if func not in self.SUPPORTED_FUNCTIONS:
                raise AggregationConfigError(
                    f"第 {i + 1} 个聚合配置的函数不支持: {func}，"
                    f"支持的函数: {', '.join(sorted(self.SUPPORTED_FUNCTIONS))}"
                )

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行聚合转换"""
        group_by = config.get("group_by", [])
        aggregations = config["aggregations"]
        having = config.get("having")
        order_by = config.get("order_by", [])

        # 构建 SQL 查询
        sql = self._build_sql(group_by, aggregations, having, order_by)

        try:
            # 使用 DuckDB 执行查询
            # 创建一个连接并注册 Polars DataFrame 作为 "data" 表
            con = duckdb.connect()
            con.register("data", data)
            result = con.execute(sql).pl()
            return result
        except Exception as e:
            raise TransformExecutionError(
                message=f"SQL 执行失败: {str(e)}",
                transformer_name=self.name,
                original_error=e,
            ) from e

    def _build_sql(
        self,
        group_by: List[str],
        aggregations: List[Dict[str, Any]],
        having: Optional[str],
        order_by: List[str],
    ) -> str:
        """构建 SQL 查询"""
        select_parts: List[str] = []

        # 添加分组字段
        select_parts.extend(group_by)

        # 添加聚合表达式
        for agg in aggregations:
            field = agg["field"]
            func = agg["function"].lower()
            alias = agg.get("alias", f"{func}_{field}")
            params = agg.get("params", {})

            agg_expr = self._build_aggregation_expr(field, func, params)
            select_parts.append(f"{agg_expr} AS {self._quote_identifier(alias)}")

        # 构建基本 SQL
        sql = f"SELECT {', '.join(select_parts)} FROM data"

        # 添加 GROUP BY
        if group_by:
            quoted_group_by = [self._quote_identifier(col) for col in group_by]
            sql += f" GROUP BY {', '.join(quoted_group_by)}"

        # 添加 HAVING
        if having:
            sql += f" HAVING {having}"

        # 添加 ORDER BY
        if order_by:
            sql += f" ORDER BY {', '.join(order_by)}"

        return sql

    def _build_aggregation_expr(
        self,
        field: str,
        func: str,
        params: Dict[str, Any],
    ) -> str:
        """构建聚合表达式"""
        quoted_field = self._quote_identifier(field)

        if func == "count":
            return f"COUNT({quoted_field})"
        elif func == "count_distinct":
            return f"COUNT(DISTINCT {quoted_field})"
        elif func == "sum":
            return f"SUM({quoted_field})"
        elif func == "avg":
            return f"AVG({quoted_field})"
        elif func == "min":
            return f"MIN({quoted_field})"
        elif func == "max":
            return f"MAX({quoted_field})"
        elif func == "first":
            return f"FIRST({quoted_field})"
        elif func == "last":
            return f"LAST({quoted_field})"
        elif func == "string_agg":
            separator = params.get("separator", ",")
            return f"STRING_AGG({quoted_field}, '{separator}')"
        elif func == "list":
            return f"LIST({quoted_field})"
        elif func == "median":
            return f"MEDIAN({quoted_field})"
        elif func == "stddev":
            return f"STDDEV({quoted_field})"
        elif func == "variance":
            return f"VARIANCE({quoted_field})"
        else:
            # 默认使用函数名
            return f"{func.upper()}({quoted_field})"

    def _quote_identifier(self, identifier: str) -> str:
        """引用标识符（处理特殊字符）"""
        # 如果标识符包含特殊字符，使用双引号包裹
        if not identifier.isidentifier() or identifier.upper() in self._reserved_words:
            return f'"{identifier}"'
        return identifier

    @property
    def _reserved_words(self) -> Set[str]:
        """DuckDB 保留字"""
        return {
            "SELECT",
            "FROM",
            "WHERE",
            "GROUP",
            "BY",
            "HAVING",
            "ORDER",
            "LIMIT",
            "OFFSET",
            "JOIN",
            "ON",
            "AND",
            "OR",
            "NOT",
            "IN",
            "IS",
            "NULL",
            "AS",
            "DISTINCT",
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
        }


@TransformerRegistry.register("duckdb_sql")
class DuckDBSQLTransformer(BaseTransformer):
    """
    DuckDB 自定义 SQL 转换器

    支持自定义 SQL 查询。

    配置示例：
        {
            "sql": "SELECT customer_id, SUM(amount) as total FROM data GROUP BY customer_id"
        }

    使用示例：
        transformer = DuckDBSQLTransformer()
        result = transformer.execute(df, {
            "sql": "SELECT * FROM data WHERE amount > 100"
        })
    """

    name: ClassVar[str] = "duckdb_sql"
    description: ClassVar[str] = "DuckDB 自定义 SQL 转换"
    version: ClassVar[str] = "1.0.0"

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证 SQL 配置"""
        if "sql" not in config:
            raise AggregationConfigError("配置缺少 'sql' 字段")

        sql = config["sql"]
        if not isinstance(sql, str) or not sql.strip():
            raise AggregationConfigError("'sql' 必须是非空字符串")

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行自定义 SQL"""
        sql = config["sql"]

        try:
            # 创建连接并注册 DataFrame
            con = duckdb.connect()
            con.register("data", data)
            result = con.execute(sql).pl()
            return result
        except Exception as e:
            raise TransformExecutionError(
                message=f"SQL 执行失败: {str(e)}",
                transformer_name=self.name,
                original_error=e,
            ) from e
