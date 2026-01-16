"""
Polars 字段映射转换器

实现 1:1 字段映射转换，支持：
- 直接映射：source -> target
- 类型转换：cast to int/float/string/datetime
- 常量值：填充固定值
- 表达式计算：简单的算术表达式
- 日期时间格式化
"""

from typing import Any, ClassVar, Dict, List, Optional, Set, Union

import polars as pl

from qdata_transformer.base import BaseTransformer
from qdata_transformer.exceptions import InvalidColumnError, MappingConfigError
from qdata_transformer.registry import TransformerRegistry


@TransformerRegistry.register("polars_field_mapping")
class PolarsFieldMappingTransformer(BaseTransformer):
    """
    Polars 字段映射转换器

    提供高性能的 1:1 字段映射能力。

    配置示例：
        {
            "mappings": [
                {"source": "order_id", "target": "id"},
                {"source": "order_date", "target": "created_at", "transform": "datetime"},
                {"source": ["qty", "price"], "target": "amount", "transform": "expression",
                 "params": {"expr": "qty * price"}},
                {"target": "status", "transform": "constant", "params": {"value": "PENDING"}}
            ],
            "drop_unmapped": false
        }

    使用示例：
        transformer = PolarsFieldMappingTransformer()
        result = transformer.execute(df, {
            "mappings": [
                {"source": "name", "target": "customer_name"},
            ]
        })
    """

    name: ClassVar[str] = "polars_field_mapping"
    description: ClassVar[str] = "Polars 1:1 字段映射转换"
    version: ClassVar[str] = "1.0.0"

    # 支持的转换类型
    TRANSFORM_TYPES: Set[Optional[str]] = {
        None,  # 直接映射
        "direct",  # 直接映射
        "constant",  # 常量值
        "cast",  # 类型转换
        "datetime",  # 日期时间格式化
        "expression",  # 表达式计算
        "rename",  # 重命名（同 direct）
    }

    # 支持的数据类型
    DTYPE_MAP: Dict[str, pl.DataType] = {
        "str": pl.Utf8,
        "string": pl.Utf8,
        "int": pl.Int64,
        "int32": pl.Int32,
        "int64": pl.Int64,
        "float": pl.Float64,
        "float32": pl.Float32,
        "float64": pl.Float64,
        "bool": pl.Boolean,
        "boolean": pl.Boolean,
        "date": pl.Date,
        "datetime": pl.Datetime,
    }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证映射配置"""
        if "mappings" not in config:
            raise MappingConfigError("配置缺少 'mappings' 字段")

        mappings = config["mappings"]
        if not isinstance(mappings, list):
            raise MappingConfigError("'mappings' 必须是列表")

        for i, mapping in enumerate(mappings):
            if not isinstance(mapping, dict):
                raise MappingConfigError(f"第 {i + 1} 个映射必须是字典")

            if "target" not in mapping:
                raise MappingConfigError(f"第 {i + 1} 个映射缺少 'target' 字段")

            transform = mapping.get("transform")
            if transform not in self.TRANSFORM_TYPES:
                raise MappingConfigError(
                    f"第 {i + 1} 个映射的 'transform' 类型不支持: {transform}",
                    field_name=mapping.get("target"),
                )

            # 验证 constant 必须有 value
            if transform == "constant":
                params = mapping.get("params", {})
                if "value" not in params:
                    raise MappingConfigError(
                        f"第 {i + 1} 个映射: constant 类型必须提供 'params.value'",
                        field_name=mapping.get("target"),
                    )

            # 验证 expression 必须有 expr
            if transform == "expression":
                params = mapping.get("params", {})
                if "expr" not in params:
                    raise MappingConfigError(
                        f"第 {i + 1} 个映射: expression 类型必须提供 'params.expr'",
                        field_name=mapping.get("target"),
                    )

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行字段映射转换"""
        mappings = config["mappings"]
        drop_unmapped = config.get("drop_unmapped", True)

        expressions: List[pl.Expr] = []
        available_columns = set(data.columns)

        for mapping in mappings:
            source = mapping.get("source")
            target = mapping["target"]
            transform_type = mapping.get("transform")
            params = mapping.get("params", {})

            expr = self._build_expression(
                source=source,
                target=target,
                transform_type=transform_type,
                params=params,
                available_columns=available_columns,
            )
            expressions.append(expr)

        if drop_unmapped:
            # 只保留映射的列
            return data.select(expressions)
        else:
            # 保留所有列，添加/替换映射的列
            return data.with_columns(expressions)

    def _build_expression(
        self,
        source: Optional[Union[str, List[str]]],
        target: str,
        transform_type: Optional[str],
        params: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """
        构建 Polars 表达式

        Args:
            source: 源字段（或字段列表）
            target: 目标字段
            transform_type: 转换类型
            params: 转换参数
            available_columns: 可用的列名集合

        Returns:
            Polars 表达式
        """
        # 常量值
        if transform_type == "constant":
            value = params["value"]
            return pl.lit(value).alias(target)

        # 无源字段且非常量，报错
        if source is None:
            raise MappingConfigError(
                f"非 constant 类型的映射必须指定 source 字段",
                field_name=target,
            )

        # 表达式计算
        if transform_type == "expression":
            if not isinstance(source, list):
                source = [source]

            # 验证源字段存在
            for col in source:
                if col not in available_columns:
                    raise InvalidColumnError(col, list(available_columns))

            expr_str = params["expr"]
            return self._parse_expression(expr_str).alias(target)

        # 单字段映射
        if isinstance(source, list):
            raise MappingConfigError(
                f"非 expression 类型只支持单个源字段",
                field_name=target,
            )

        # 验证源字段存在
        if source not in available_columns:
            raise InvalidColumnError(source, list(available_columns))

        expr = pl.col(source)

        # 直接映射或重命名
        if transform_type in (None, "direct", "rename"):
            return expr.alias(target)

        # 类型转换
        if transform_type == "cast":
            dtype_str = params.get("dtype", "str")
            dtype = self.DTYPE_MAP.get(dtype_str)
            if dtype is None:
                raise MappingConfigError(
                    f"不支持的数据类型: {dtype_str}",
                    field_name=target,
                )
            return expr.cast(dtype).alias(target)

        # 日期时间格式化
        if transform_type == "datetime":
            input_format = params.get("format", "%Y-%m-%d")
            return expr.str.strptime(pl.Datetime, input_format).alias(target)

        # 不应该到达这里
        return expr.alias(target)

    def _parse_expression(self, expr_str: str) -> pl.Expr:
        """
        解析简单表达式

        支持的表达式：
        - 简单算术：col1 * col2, col1 + col2, col1 - col2, col1 / col2
        - 字符串拼接：col1 + "_" + col2（当操作数包含引号）

        Args:
            expr_str: 表达式字符串

        Returns:
            Polars 表达式
        """
        expr_str = expr_str.strip()

        # 尝试解析二元运算
        for op in ["*", "+", "-", "/"]:
            if op in expr_str:
                parts = expr_str.split(op, 1)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()

                    # 验证操作数非空
                    if not left or not right:
                        continue

                    left_expr = self._parse_operand(left)
                    right_expr = self._parse_operand(right)

                    if op == "*":
                        return left_expr * right_expr
                    elif op == "+":
                        return left_expr + right_expr
                    elif op == "-":
                        return left_expr - right_expr
                    elif op == "/":
                        return left_expr / right_expr

        # 单列引用
        return pl.col(expr_str)

    def _parse_operand(self, operand: str) -> pl.Expr:
        """
        解析操作数

        Args:
            operand: 操作数字符串

        Returns:
            Polars 表达式
        """
        operand = operand.strip()

        # 字符串字面量
        if (operand.startswith("'") and operand.endswith("'")) or (
            operand.startswith('"') and operand.endswith('"')
        ):
            return pl.lit(operand[1:-1])

        # 数字字面量
        try:
            if "." in operand:
                return pl.lit(float(operand))
            else:
                return pl.lit(int(operand))
        except ValueError:
            pass

        # 列引用
        return pl.col(operand)
