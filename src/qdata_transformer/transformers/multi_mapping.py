"""
Polars 批量映射转换器

实现 1N:1N 批量数据映射转换，支持：
- 批量字段映射：一次性处理多行数据
- 嵌套字段展开：处理嵌套的 JSON 结构
- 数组字段展开：将数组字段展开为多行
- 条件过滤：在映射过程中过滤数据
"""

from typing import Any, ClassVar, Dict, List, Optional, Set, Union

import polars as pl

from qdata_transformer.base import BaseTransformer
from qdata_transformer.exceptions import InvalidColumnError, MappingConfigError
from qdata_transformer.registry import TransformerRegistry


@TransformerRegistry.register("polars_multi_mapping")
class PolarsMultiMappingTransformer(BaseTransformer):
    """
    Polars 批量映射转换器

    提供高性能的 1N:1N 批量映射能力，适用于批量数据转换场景。

    配置示例：
        {
            "mappings": [
                {"source": "order_id", "target": "id"},
                {"source": "customer.name", "target": "customer_name"},
                {"source": "items", "target": "item_list", "transform": "explode"}
            ],
            "filter": {
                "condition": "amount > 0"
            },
            "batch_size": 1000
        }

    使用示例：
        transformer = PolarsMultiMappingTransformer()
        result = transformer.execute(df, {
            "mappings": [
                {"source": "name", "target": "customer_name"},
            ]
        })
    """

    name: ClassVar[str] = "polars_multi_mapping"
    description: ClassVar[str] = "Polars 1N:1N 批量映射转换"
    version: ClassVar[str] = "1.0.0"

    # 支持的转换类型
    TRANSFORM_TYPES: Set[Optional[str]] = {
        None,  # 直接映射
        "direct",  # 直接映射
        "nested",  # 嵌套字段访问
        "explode",  # 数组展开
        "constant",  # 常量值
        "cast",  # 类型转换
        "coalesce",  # 合并空值（取第一个非空值）
        "concat",  # 字符串连接
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

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行批量映射转换"""
        mappings = config["mappings"]
        filter_config = config.get("filter")
        drop_unmapped = config.get("drop_unmapped", True)

        result = data
        available_columns = set(data.columns)
        expressions: List[pl.Expr] = []
        explode_columns: List[str] = []

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

            # 记录需要 explode 的列
            if transform_type == "explode":
                explode_columns.append(target)

        # 应用映射
        if drop_unmapped:
            result = result.select(expressions)
        else:
            result = result.with_columns(expressions)

        # 展开数组列
        for col in explode_columns:
            result = result.explode(col)

        # 应用过滤条件
        if filter_config and "condition" in filter_config:
            result = self._apply_filter(result, filter_config["condition"])

        return result

    def _build_expression(
        self,
        source: Optional[Union[str, List[str]]],
        target: str,
        transform_type: Optional[str],
        params: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建 Polars 表达式"""
        # 常量值
        if transform_type == "constant":
            value = params.get("value")
            return pl.lit(value).alias(target)

        # 无源字段且非常量，报错
        if source is None:
            raise MappingConfigError(
                f"非 constant 类型的映射必须指定 source 字段",
                field_name=target,
            )

        # 嵌套字段访问
        if transform_type == "nested":
            return self._handle_nested_field(source, target, available_columns)

        # 合并空值 (coalesce)
        if transform_type == "coalesce":
            if not isinstance(source, list):
                source = [source]
            return self._handle_coalesce(source, target, available_columns)

        # 字符串连接
        if transform_type == "concat":
            if not isinstance(source, list):
                source = [source]
            separator = params.get("separator", "")
            return self._handle_concat(source, target, separator, available_columns)

        # 单字段处理
        if isinstance(source, list):
            # 对于 explode 等类型，使用第一个字段
            source = source[0]

        # 验证源字段存在（对于嵌套字段，验证根字段）
        root_col = source.split(".")[0] if "." in source else source
        if root_col not in available_columns:
            raise InvalidColumnError(root_col, list(available_columns))

        expr = pl.col(source)

        # 直接映射
        if transform_type in (None, "direct"):
            return expr.alias(target)

        # 数组展开（先添加别名，后续会 explode）
        if transform_type == "explode":
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

        return expr.alias(target)

    def _handle_nested_field(
        self,
        source: Union[str, List[str]],
        target: str,
        available_columns: Set[str],
    ) -> pl.Expr:
        """
        处理嵌套字段

        支持点号分隔的嵌套字段访问，如 "customer.name"。
        """
        if isinstance(source, list):
            source = source[0]

        parts = source.split(".")
        root_col = parts[0]

        if root_col not in available_columns:
            raise InvalidColumnError(root_col, list(available_columns))

        # 构建嵌套访问表达式
        expr = pl.col(root_col)
        for part in parts[1:]:
            expr = expr.struct.field(part)

        return expr.alias(target)

    def _handle_coalesce(
        self,
        sources: List[str],
        target: str,
        available_columns: Set[str],
    ) -> pl.Expr:
        """处理 coalesce（取第一个非空值）"""
        exprs = []
        for source in sources:
            if source not in available_columns:
                raise InvalidColumnError(source, list(available_columns))
            exprs.append(pl.col(source))

        return pl.coalesce(exprs).alias(target)

    def _handle_concat(
        self,
        sources: List[str],
        target: str,
        separator: str,
        available_columns: Set[str],
    ) -> pl.Expr:
        """处理字符串连接"""
        exprs = []
        for source in sources:
            if source not in available_columns:
                raise InvalidColumnError(source, list(available_columns))
            exprs.append(pl.col(source).cast(pl.Utf8))

        return pl.concat_str(exprs, separator=separator).alias(target)

    def _apply_filter(self, data: pl.DataFrame, condition: str) -> pl.DataFrame:
        """
        应用过滤条件

        支持简单的过滤表达式：
        - column > value
        - column == value
        - column != value
        - column < value
        - column >= value
        - column <= value
        """
        condition = condition.strip()

        # 解析比较运算符
        operators = [">=", "<=", "!=", "==", ">", "<"]

        for op in operators:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    col_name = parts[0].strip()
                    value_str = parts[1].strip()

                    # 解析值
                    value = self._parse_value(value_str)

                    # 构建过滤表达式
                    col_expr = pl.col(col_name)

                    if op == ">":
                        return data.filter(col_expr > value)
                    elif op == "<":
                        return data.filter(col_expr < value)
                    elif op == ">=":
                        return data.filter(col_expr >= value)
                    elif op == "<=":
                        return data.filter(col_expr <= value)
                    elif op == "==":
                        return data.filter(col_expr == value)
                    elif op == "!=":
                        return data.filter(col_expr != value)

        return data

    def _parse_value(self, value_str: str) -> Any:
        """解析值字符串"""
        value_str = value_str.strip()

        # 字符串字面量
        if (value_str.startswith("'") and value_str.endswith("'")) or (
            value_str.startswith('"') and value_str.endswith('"')
        ):
            return value_str[1:-1]

        # 布尔值
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # None/null
        if value_str.lower() in ("none", "null"):
            return None

        # 数字
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # 默认作为字符串
        return value_str
