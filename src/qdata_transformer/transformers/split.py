"""
Polars 数据拆分转换器

实现高性能的数据拆分能力，支持：
- 数组展开：将数组列展开为多行
- 字符串拆分：按分隔符拆分字符串为多行
- JSON 展开：展开 JSON 结构
"""

from typing import Any, ClassVar, Dict, List, Optional, Set

import polars as pl

from qdata_transformer.base import BaseTransformer
from qdata_transformer.exceptions import TransformerConfigError, InvalidColumnError
from qdata_transformer.registry import TransformerRegistry


@TransformerRegistry.register("polars_split")
class PolarsSplitTransformer(BaseTransformer):
    """
    Polars 数据拆分转换器

    提供高性能的数据拆分能力。

    配置示例：
        {
            "column": "items",
            "type": "explode"
        }

        或者字符串拆分：
        {
            "column": "tags",
            "type": "split",
            "separator": ","
        }

    支持的拆分类型：
        - explode: 展开数组/列表列
        - split: 按分隔符拆分字符串
        - unnest: 展开结构体列

    使用示例：
        transformer = PolarsSplitTransformer()
        result = transformer.execute(df, {
            "column": "items",
            "type": "explode"
        })
    """

    name: ClassVar[str] = "polars_split"
    description: ClassVar[str] = "Polars 数据拆分转换"
    version: ClassVar[str] = "1.0.0"

    # 支持的拆分类型
    SUPPORTED_TYPES: Set[str] = {"explode", "split", "unnest"}

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证拆分配置"""
        if "column" not in config:
            raise TransformerConfigError("配置缺少 'column' 字段")

        split_type = config.get("type", "explode")
        if split_type not in self.SUPPORTED_TYPES:
            raise TransformerConfigError(
                f"拆分类型不支持: {split_type}，"
                f"支持的类型: {', '.join(sorted(self.SUPPORTED_TYPES))}"
            )

        # split 类型需要 separator
        if split_type == "split" and "separator" not in config:
            raise TransformerConfigError("'split' 类型需要 'separator' 参数")

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行数据拆分"""
        column = config["column"]
        split_type = config.get("type", "explode")

        available_columns = set(data.columns)
        if column not in available_columns:
            raise InvalidColumnError(column, list(available_columns))

        if split_type == "explode":
            return self._explode(data, column)
        elif split_type == "split":
            separator = config["separator"]
            return self._split(data, column, separator)
        elif split_type == "unnest":
            return self._unnest(data, column)
        else:
            raise TransformerConfigError(f"不支持的拆分类型: {split_type}")

    def _explode(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        """展开数组列"""
        return data.explode(column)

    def _split(
        self,
        data: pl.DataFrame,
        column: str,
        separator: str,
    ) -> pl.DataFrame:
        """按分隔符拆分字符串并展开"""
        return data.with_columns(
            pl.col(column).str.split(separator)
        ).explode(column)

    def _unnest(self, data: pl.DataFrame, column: str) -> pl.DataFrame:
        """展开结构体列"""
        return data.unnest(column)


@TransformerRegistry.register("polars_deduplicate")
class PolarsDeduplicateTransformer(BaseTransformer):
    """
    Polars 数据去重转换器

    提供高性能的数据去重能力。

    配置示例：
        {
            "columns": ["customer_id", "order_date"],
            "keep": "first",
            "maintain_order": true
        }

    参数说明：
        - columns: 去重依据的列（可选，为空时对所有列去重）
        - keep: 保留策略 ("first", "last", "none")
        - maintain_order: 是否保持原始顺序

    使用示例：
        transformer = PolarsDeduplicateTransformer()
        result = transformer.execute(df, {
            "columns": ["id"],
            "keep": "first"
        })
    """

    name: ClassVar[str] = "polars_deduplicate"
    description: ClassVar[str] = "Polars 数据去重转换"
    version: ClassVar[str] = "1.0.0"

    # 支持的保留策略
    KEEP_STRATEGIES: Set[str] = {"first", "last", "none", "any"}

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证去重配置"""
        columns = config.get("columns")
        if columns is not None and not isinstance(columns, list):
            raise TransformerConfigError("'columns' 必须是列表或为空")

        keep = config.get("keep", "first")
        if keep not in self.KEEP_STRATEGIES:
            raise TransformerConfigError(
                f"'keep' 参数不支持: {keep}，"
                f"支持的值: {', '.join(sorted(self.KEEP_STRATEGIES))}"
            )

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行数据去重"""
        columns = config.get("columns")
        keep = config.get("keep", "first")
        maintain_order = config.get("maintain_order", True)

        # 验证列存在
        if columns:
            available_columns = set(data.columns)
            for col in columns:
                if col not in available_columns:
                    raise InvalidColumnError(col, list(available_columns))

        # 执行去重
        if columns:
            subset = columns
        else:
            subset = None

        return data.unique(
            subset=subset,
            keep=keep,
            maintain_order=maintain_order,
        )


@TransformerRegistry.register("polars_merge")
class PolarsMergeTransformer(BaseTransformer):
    """
    Polars 数据合并转换器

    提供高性能的数据合并能力（垂直合并）。

    配置示例：
        {
            "how": "vertical",
            "rechunk": true
        }

    注意：此转换器用于合并单个 DataFrame 中的列或行，
    对于两个 DataFrame 的合并，请使用 DuckDB JOIN 转换器。

    使用示例：
        transformer = PolarsMergeTransformer()
        result = transformer.execute(df, {"how": "vertical"})
    """

    name: ClassVar[str] = "polars_merge"
    description: ClassVar[str] = "Polars 数据合并转换"
    version: ClassVar[str] = "1.0.0"

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证合并配置"""
        # 此转换器配置简单，主要验证参数有效性
        how = config.get("how", "vertical")
        if how not in ("vertical", "diagonal"):
            raise TransformerConfigError(f"'how' 参数不支持: {how}")

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行数据合并（对单个 DataFrame 的处理）"""
        # 对于单个 DataFrame，此转换器主要用于 rechunk 优化
        rechunk = config.get("rechunk", False)
        if rechunk:
            return data.rechunk()
        return data
