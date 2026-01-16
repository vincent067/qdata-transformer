"""
Polars 数据过滤转换器

实现高性能的数据过滤能力，支持：
- 条件过滤：基于表达式的行过滤
- 空值过滤：过滤包含空值的行
- 范围过滤：数值/日期范围过滤
- 正则过滤：基于正则表达式的字符串过滤
"""

from typing import Any, ClassVar, Dict, List, Optional, Set

import polars as pl

from qdata_transformer.base import BaseTransformer
from qdata_transformer.exceptions import TransformerConfigError, InvalidColumnError
from qdata_transformer.registry import TransformerRegistry


@TransformerRegistry.register("polars_filter")
class PolarsFilterTransformer(BaseTransformer):
    """
    Polars 数据过滤转换器

    提供高性能的数据过滤能力。

    配置示例：
        {
            "conditions": [
                {"column": "amount", "operator": ">", "value": 100},
                {"column": "status", "operator": "==", "value": "active"},
                {"column": "name", "operator": "contains", "value": "test"}
            ],
            "logic": "and",
            "drop_null": ["amount", "status"]
        }

    支持的操作符：
        - ==, !=: 相等/不相等
        - >, <, >=, <=: 比较
        - in, not_in: 包含/不包含
        - contains, starts_with, ends_with: 字符串匹配
        - regex: 正则匹配
        - is_null, is_not_null: 空值检查
        - between: 范围检查

    使用示例：
        transformer = PolarsFilterTransformer()
        result = transformer.execute(df, {
            "conditions": [
                {"column": "amount", "operator": ">", "value": 0}
            ]
        })
    """

    name: ClassVar[str] = "polars_filter"
    description: ClassVar[str] = "Polars 数据过滤转换"
    version: ClassVar[str] = "1.0.0"

    # 支持的操作符
    SUPPORTED_OPERATORS: Set[str] = {
        "==",
        "!=",
        ">",
        "<",
        ">=",
        "<=",
        "in",
        "not_in",
        "contains",
        "starts_with",
        "ends_with",
        "regex",
        "is_null",
        "is_not_null",
        "between",
    }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证过滤配置"""
        conditions = config.get("conditions", [])
        if not isinstance(conditions, list):
            raise TransformerConfigError("'conditions' 必须是列表")

        for i, cond in enumerate(conditions):
            if not isinstance(cond, dict):
                raise TransformerConfigError(f"第 {i + 1} 个条件必须是字典")

            if "column" not in cond:
                raise TransformerConfigError(f"第 {i + 1} 个条件缺少 'column' 字段")

            if "operator" not in cond:
                raise TransformerConfigError(f"第 {i + 1} 个条件缺少 'operator' 字段")

            operator = cond["operator"]
            if operator not in self.SUPPORTED_OPERATORS:
                raise TransformerConfigError(
                    f"第 {i + 1} 个条件的操作符不支持: {operator}，"
                    f"支持的操作符: {', '.join(sorted(self.SUPPORTED_OPERATORS))}"
                )

            # 验证特定操作符需要的参数
            if operator in ("in", "not_in"):
                if "value" not in cond or not isinstance(cond["value"], list):
                    raise TransformerConfigError(
                        f"第 {i + 1} 个条件: '{operator}' 操作符需要 'value' 为列表"
                    )

            if operator == "between":
                if "min" not in cond or "max" not in cond:
                    raise TransformerConfigError(
                        f"第 {i + 1} 个条件: 'between' 操作符需要 'min' 和 'max' 参数"
                    )

            if operator not in ("is_null", "is_not_null", "between") and "value" not in cond:
                raise TransformerConfigError(
                    f"第 {i + 1} 个条件缺少 'value' 字段"
                )

        logic = config.get("logic", "and")
        if logic not in ("and", "or"):
            raise TransformerConfigError(f"'logic' 必须是 'and' 或 'or'，收到: {logic}")

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行数据过滤"""
        conditions = config.get("conditions", [])
        logic = config.get("logic", "and")
        drop_null = config.get("drop_null", [])

        result = data
        available_columns = set(data.columns)

        # 首先处理空值过滤
        if drop_null:
            for col in drop_null:
                if col not in available_columns:
                    raise InvalidColumnError(col, list(available_columns))
                result = result.filter(pl.col(col).is_not_null())

        # 处理条件过滤
        if conditions:
            filter_exprs: List[pl.Expr] = []

            for cond in conditions:
                column = cond["column"]
                if column not in available_columns:
                    raise InvalidColumnError(column, list(available_columns))

                expr = self._build_filter_expr(cond)
                filter_exprs.append(expr)

            if filter_exprs:
                if logic == "and":
                    combined_expr = filter_exprs[0]
                    for expr in filter_exprs[1:]:
                        combined_expr = combined_expr & expr
                else:  # or
                    combined_expr = filter_exprs[0]
                    for expr in filter_exprs[1:]:
                        combined_expr = combined_expr | expr

                result = result.filter(combined_expr)

        return result

    def _build_filter_expr(self, condition: Dict[str, Any]) -> pl.Expr:
        """构建过滤表达式"""
        column = condition["column"]
        operator = condition["operator"]
        value = condition.get("value")

        col_expr = pl.col(column)

        if operator == "==":
            return col_expr == value
        elif operator == "!=":
            return col_expr != value
        elif operator == ">":
            return col_expr > value
        elif operator == "<":
            return col_expr < value
        elif operator == ">=":
            return col_expr >= value
        elif operator == "<=":
            return col_expr <= value
        elif operator == "in":
            return col_expr.is_in(value)
        elif operator == "not_in":
            return ~col_expr.is_in(value)
        elif operator == "contains":
            return col_expr.str.contains(str(value))
        elif operator == "starts_with":
            return col_expr.str.starts_with(str(value))
        elif operator == "ends_with":
            return col_expr.str.ends_with(str(value))
        elif operator == "regex":
            return col_expr.str.contains(str(value))
        elif operator == "is_null":
            return col_expr.is_null()
        elif operator == "is_not_null":
            return col_expr.is_not_null()
        elif operator == "between":
            min_val = condition["min"]
            max_val = condition["max"]
            return (col_expr >= min_val) & (col_expr <= max_val)
        else:
            raise TransformerConfigError(f"不支持的操作符: {operator}")
