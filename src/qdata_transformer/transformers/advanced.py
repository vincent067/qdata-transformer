"""
DuckDB JOIN 和窗口函数转换器

实现基于 DuckDB 的高级 SQL 功能，支持：
- JOIN 操作：INNER, LEFT, RIGHT, FULL, CROSS JOIN
- 窗口函数：ROW_NUMBER, RANK, DENSE_RANK, LEAD, LAG, SUM OVER, AVG OVER 等
"""

from typing import Any, ClassVar, Dict, List, Optional, Set

import duckdb
import polars as pl

from qdata_transformer.base import BaseTransformer
from qdata_transformer.exceptions import TransformerConfigError, TransformExecutionError
from qdata_transformer.registry import TransformerRegistry


@TransformerRegistry.register("duckdb_join")
class DuckDBJoinTransformer(BaseTransformer):
    """
    DuckDB JOIN 转换器

    提供高性能的 SQL JOIN 能力。

    注意：由于转换器设计为单输入，此转换器需要在 config 中提供
    第二个数据集，或者通过 context 参数传入。

    配置示例：
        {
            "right_data": [...],  # 或通过 context 提供
            "join_type": "inner",
            "left_on": ["customer_id"],
            "right_on": ["id"],
            "suffix": "_right"
        }

    支持的 JOIN 类型：
        - inner: 内连接
        - left: 左连接
        - right: 右连接
        - full: 全连接
        - cross: 交叉连接

    使用示例：
        transformer = DuckDBJoinTransformer()
        result = transformer.execute(df, {
            "right_data": [{"id": 1, "name": "A"}],
            "join_type": "inner",
            "left_on": ["customer_id"],
            "right_on": ["id"]
        })
    """

    name: ClassVar[str] = "duckdb_join"
    description: ClassVar[str] = "DuckDB SQL JOIN 转换"
    version: ClassVar[str] = "1.0.0"

    # 支持的 JOIN 类型
    SUPPORTED_JOIN_TYPES: Set[str] = {"inner", "left", "right", "full", "cross"}

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证 JOIN 配置"""
        join_type = config.get("join_type", "inner")
        if join_type not in self.SUPPORTED_JOIN_TYPES:
            raise TransformerConfigError(
                f"JOIN 类型不支持: {join_type}，"
                f"支持的类型: {', '.join(sorted(self.SUPPORTED_JOIN_TYPES))}"
            )

        # 非 cross join 需要 on 条件
        if join_type != "cross":
            if "left_on" not in config or "right_on" not in config:
                raise TransformerConfigError(
                    f"'{join_type}' JOIN 需要 'left_on' 和 'right_on' 参数"
                )

            left_on = config["left_on"]
            right_on = config["right_on"]

            if not isinstance(left_on, list) or not isinstance(right_on, list):
                raise TransformerConfigError(
                    "'left_on' 和 'right_on' 必须是列表"
                )

            if len(left_on) != len(right_on):
                raise TransformerConfigError(
                    "'left_on' 和 'right_on' 长度必须相同"
                )

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行 JOIN 转换"""
        join_type = config.get("join_type", "inner")
        left_on = config.get("left_on", [])
        right_on = config.get("right_on", [])
        suffix = config.get("suffix", "_right")

        # 获取右表数据
        right_data = config.get("right_data")
        if right_data is None:
            raise TransformerConfigError("JOIN 操作需要 'right_data' 参数")

        # 转换为 Polars DataFrame
        if isinstance(right_data, list):
            right_df = pl.DataFrame(right_data)
        elif isinstance(right_data, pl.DataFrame):
            right_df = right_data
        else:
            raise TransformerConfigError(
                "'right_data' 必须是列表或 Polars DataFrame"
            )

        # 构建并执行 SQL
        sql = self._build_join_sql(join_type, left_on, right_on, suffix, right_df.columns)

        try:
            con = duckdb.connect()
            con.register("left_table", data)
            con.register("right_table", right_df)
            result = con.execute(sql).pl()
            return result
        except Exception as e:
            raise TransformExecutionError(
                message=f"JOIN 执行失败: {str(e)}",
                transformer_name=self.name,
                original_error=e,
            ) from e

    def _build_join_sql(
        self,
        join_type: str,
        left_on: List[str],
        right_on: List[str],
        suffix: str,
        right_columns: List[str],
    ) -> str:
        """构建 JOIN SQL"""
        join_type_map = {
            "inner": "INNER JOIN",
            "left": "LEFT JOIN",
            "right": "RIGHT JOIN",
            "full": "FULL OUTER JOIN",
            "cross": "CROSS JOIN",
        }

        sql_join_type = join_type_map[join_type]

        # 构建 SELECT 子句（处理列名冲突）
        select_parts = ["left_table.*"]
        for col in right_columns:
            if col in right_on:
                continue  # JOIN 键不重复选择
            select_parts.append(f"right_table.{self._quote(col)} AS {self._quote(col + suffix)}")

        sql = f"SELECT {', '.join(select_parts)} FROM left_table {sql_join_type} right_table"

        # 添加 ON 条件
        if join_type != "cross":
            on_conditions = []
            for left_col, right_col in zip(left_on, right_on):
                on_conditions.append(
                    f"left_table.{self._quote(left_col)} = right_table.{self._quote(right_col)}"
                )
            sql += f" ON {' AND '.join(on_conditions)}"

        return sql

    def _quote(self, identifier: str) -> str:
        """引用标识符"""
        if not identifier.isidentifier():
            return f'"{identifier}"'
        return identifier


@TransformerRegistry.register("duckdb_window")
class DuckDBWindowTransformer(BaseTransformer):
    """
    DuckDB 窗口函数转换器

    提供高性能的 SQL 窗口函数能力。

    配置示例：
        {
            "functions": [
                {
                    "function": "row_number",
                    "alias": "rn",
                    "partition_by": ["customer_id"],
                    "order_by": ["order_date DESC"]
                },
                {
                    "function": "sum",
                    "column": "amount",
                    "alias": "running_total",
                    "partition_by": ["customer_id"],
                    "order_by": ["order_date"],
                    "frame": "ROWS UNBOUNDED PRECEDING"
                },
                {
                    "function": "lag",
                    "column": "amount",
                    "alias": "prev_amount",
                    "partition_by": ["customer_id"],
                    "order_by": ["order_date"],
                    "offset": 1,
                    "default": 0
                }
            ]
        }

    支持的窗口函数：
        排名函数：
        - row_number: 行号
        - rank: 排名（有间隙）
        - dense_rank: 密集排名（无间隙）
        - ntile: N 分位
        - percent_rank: 百分比排名
        - cume_dist: 累计分布

        偏移函数：
        - lag: 前 N 行
        - lead: 后 N 行
        - first_value: 第一个值
        - last_value: 最后一个值
        - nth_value: 第 N 个值

        聚合函数（窗口版）：
        - sum: 累计求和
        - avg: 滑动平均
        - min: 滑动最小
        - max: 滑动最大
        - count: 滑动计数

    使用示例：
        transformer = DuckDBWindowTransformer()
        result = transformer.execute(df, {
            "functions": [
                {"function": "row_number", "alias": "rn", "order_by": ["id"]}
            ]
        })
    """

    name: ClassVar[str] = "duckdb_window"
    description: ClassVar[str] = "DuckDB SQL 窗口函数转换"
    version: ClassVar[str] = "1.0.0"

    # 支持的窗口函数
    RANKING_FUNCTIONS: Set[str] = {
        "row_number",
        "rank",
        "dense_rank",
        "ntile",
        "percent_rank",
        "cume_dist",
    }

    OFFSET_FUNCTIONS: Set[str] = {
        "lag",
        "lead",
        "first_value",
        "last_value",
        "nth_value",
    }

    AGGREGATE_FUNCTIONS: Set[str] = {
        "sum",
        "avg",
        "min",
        "max",
        "count",
    }

    @property
    def supported_functions(self) -> Set[str]:
        return self.RANKING_FUNCTIONS | self.OFFSET_FUNCTIONS | self.AGGREGATE_FUNCTIONS

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证窗口函数配置"""
        if "functions" not in config:
            raise TransformerConfigError("配置缺少 'functions' 字段")

        functions = config["functions"]
        if not isinstance(functions, list):
            raise TransformerConfigError("'functions' 必须是列表")

        for i, func in enumerate(functions):
            if not isinstance(func, dict):
                raise TransformerConfigError(f"第 {i + 1} 个函数配置必须是字典")

            if "function" not in func:
                raise TransformerConfigError(f"第 {i + 1} 个函数配置缺少 'function' 字段")

            func_name = func["function"].lower()
            if func_name not in self.supported_functions:
                raise TransformerConfigError(
                    f"第 {i + 1} 个函数不支持: {func_name}，"
                    f"支持的函数: {', '.join(sorted(self.supported_functions))}"
                )

            if "alias" not in func:
                raise TransformerConfigError(f"第 {i + 1} 个函数配置缺少 'alias' 字段")

            # 聚合和偏移函数需要 column 参数
            if func_name in (self.AGGREGATE_FUNCTIONS | self.OFFSET_FUNCTIONS):
                if func_name not in ("count",) and "column" not in func:
                    raise TransformerConfigError(
                        f"第 {i + 1} 个函数 '{func_name}' 需要 'column' 参数"
                    )

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行窗口函数转换"""
        functions = config["functions"]

        # 构建 SQL
        sql = self._build_window_sql(functions)

        try:
            con = duckdb.connect()
            con.register("data", data)
            result = con.execute(sql).pl()
            return result
        except Exception as e:
            raise TransformExecutionError(
                message=f"窗口函数执行失败: {str(e)}",
                transformer_name=self.name,
                original_error=e,
            ) from e

    def _build_window_sql(self, functions: List[Dict[str, Any]]) -> str:
        """构建窗口函数 SQL"""
        select_parts = ["*"]

        for func in functions:
            func_name = func["function"].lower()
            alias = func["alias"]
            partition_by = func.get("partition_by", [])
            order_by = func.get("order_by", [])
            frame = func.get("frame")
            column = func.get("column")

            # 构建窗口表达式
            window_expr = self._build_window_expr(
                func_name, column, partition_by, order_by, frame, func
            )

            select_parts.append(f"{window_expr} AS {self._quote(alias)}")

        return f"SELECT {', '.join(select_parts)} FROM data"

    def _build_window_expr(
        self,
        func_name: str,
        column: Optional[str],
        partition_by: List[str],
        order_by: List[str],
        frame: Optional[str],
        config: Dict[str, Any],
    ) -> str:
        """构建单个窗口函数表达式"""
        # 构建函数调用部分
        if func_name in self.RANKING_FUNCTIONS:
            if func_name == "ntile":
                n = config.get("n", 4)
                func_call = f"NTILE({n})"
            else:
                func_call = f"{func_name.upper()}()"

        elif func_name in self.OFFSET_FUNCTIONS:
            quoted_col = self._quote(column) if column else "*"
            if func_name in ("lag", "lead"):
                offset = config.get("offset", 1)
                default = config.get("default")
                if default is not None:
                    func_call = f"{func_name.upper()}({quoted_col}, {offset}, {self._format_value(default)})"
                else:
                    func_call = f"{func_name.upper()}({quoted_col}, {offset})"
            elif func_name == "nth_value":
                n = config.get("n", 1)
                func_call = f"NTH_VALUE({quoted_col}, {n})"
            else:
                func_call = f"{func_name.upper()}({quoted_col})"

        elif func_name in self.AGGREGATE_FUNCTIONS:
            if func_name == "count" and not column:
                func_call = "COUNT(*)"
            else:
                quoted_col = self._quote(column) if column else "*"
                func_call = f"{func_name.upper()}({quoted_col})"

        else:
            func_call = f"{func_name.upper()}()"

        # 构建 OVER 子句
        over_parts = []

        if partition_by:
            quoted_partition = [self._quote(col) for col in partition_by]
            over_parts.append(f"PARTITION BY {', '.join(quoted_partition)}")

        if order_by:
            over_parts.append(f"ORDER BY {', '.join(order_by)}")

        if frame:
            over_parts.append(frame)

        over_clause = " ".join(over_parts) if over_parts else ""
        return f"{func_call} OVER ({over_clause})"

    def _quote(self, identifier: str) -> str:
        """引用标识符"""
        if not identifier.isidentifier():
            return f'"{identifier}"'
        return identifier

    def _format_value(self, value: Any) -> str:
        """格式化值"""
        if value is None:
            return "NULL"
        elif isinstance(value, str):
            return f"'{value}'"
        else:
            return str(value)
