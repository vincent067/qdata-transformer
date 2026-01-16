"""
Polars 列运算转换器

实现高性能的列运算能力，支持：
- 算术运算：加减乘除
- 字符串运算：连接、替换、大小写转换
- 日期运算：日期加减、提取年月日等
- 逻辑运算：与或非
- 条件运算：when/then/otherwise
"""

from typing import Any, ClassVar, Dict, List, Optional, Set, Union

import polars as pl

from qdata_transformer.base import BaseTransformer
from qdata_transformer.exceptions import TransformerConfigError, InvalidColumnError
from qdata_transformer.registry import TransformerRegistry


@TransformerRegistry.register("polars_column_ops")
class PolarsColumnOpsTransformer(BaseTransformer):
    """
    Polars 列运算转换器

    提供高性能的列运算能力。

    配置示例：
        {
            "operations": [
                {
                    "type": "arithmetic",
                    "column": "quantity",
                    "operator": "*",
                    "operand": "price",
                    "result": "amount"
                },
                {
                    "type": "string",
                    "column": "name",
                    "operation": "upper",
                    "result": "name_upper"
                },
                {
                    "type": "conditional",
                    "when": {"column": "amount", "operator": ">", "value": 1000},
                    "then": "high",
                    "otherwise": "low",
                    "result": "level"
                }
            ]
        }

    支持的运算类型：
        - arithmetic: 算术运算 (+, -, *, /, //, %)
        - string: 字符串运算 (upper, lower, strip, replace, concat, split, etc.)
        - datetime: 日期运算 (extract_year, extract_month, add_days, etc.)
        - conditional: 条件运算 (when/then/otherwise)
        - fill: 填充运算 (fill_null, fill_nan, forward_fill, backward_fill)
        - round: 数值四舍五入
        - abs: 绝对值
        - clip: 裁剪到范围

    使用示例：
        transformer = PolarsColumnOpsTransformer()
        result = transformer.execute(df, {
            "operations": [
                {"type": "arithmetic", "column": "a", "operator": "+", "operand": 10, "result": "a_plus_10"}
            ]
        })
    """

    name: ClassVar[str] = "polars_column_ops"
    description: ClassVar[str] = "Polars 列运算转换"
    version: ClassVar[str] = "1.0.0"

    # 支持的运算类型
    SUPPORTED_TYPES: Set[str] = {
        "arithmetic",
        "string",
        "datetime",
        "conditional",
        "fill",
        "round",
        "abs",
        "clip",
    }

    # 支持的算术操作符
    ARITHMETIC_OPERATORS: Set[str] = {"+", "-", "*", "/", "//", "%"}

    # 支持的字符串操作
    STRING_OPERATIONS: Set[str] = {
        "upper",
        "lower",
        "strip",
        "lstrip",
        "rstrip",
        "replace",
        "concat",
        "slice",
        "length",
        "reverse",
        "capitalize",
    }

    # 支持的日期操作
    DATETIME_OPERATIONS: Set[str] = {
        "extract_year",
        "extract_month",
        "extract_day",
        "extract_hour",
        "extract_minute",
        "extract_second",
        "extract_weekday",
        "extract_week",
        "add_days",
        "add_months",
        "truncate_day",
        "truncate_month",
    }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证列运算配置"""
        if "operations" not in config:
            raise TransformerConfigError("配置缺少 'operations' 字段")

        operations = config["operations"]
        if not isinstance(operations, list):
            raise TransformerConfigError("'operations' 必须是列表")

        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                raise TransformerConfigError(f"第 {i + 1} 个操作必须是字典")

            if "type" not in op:
                raise TransformerConfigError(f"第 {i + 1} 个操作缺少 'type' 字段")

            op_type = op["type"]
            if op_type not in self.SUPPORTED_TYPES:
                raise TransformerConfigError(
                    f"第 {i + 1} 个操作的类型不支持: {op_type}，"
                    f"支持的类型: {', '.join(sorted(self.SUPPORTED_TYPES))}"
                )

            if "result" not in op:
                raise TransformerConfigError(f"第 {i + 1} 个操作缺少 'result' 字段")

    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """执行列运算"""
        operations = config["operations"]
        available_columns = set(data.columns)

        expressions: List[pl.Expr] = []

        for op in operations:
            op_type = op["type"]
            result_name = op["result"]

            expr = self._build_operation_expr(op, available_columns)
            expressions.append(expr.alias(result_name))

        return data.with_columns(expressions)

    def _build_operation_expr(
        self,
        operation: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建运算表达式"""
        op_type = operation["type"]

        if op_type == "arithmetic":
            return self._build_arithmetic_expr(operation, available_columns)
        elif op_type == "string":
            return self._build_string_expr(operation, available_columns)
        elif op_type == "datetime":
            return self._build_datetime_expr(operation, available_columns)
        elif op_type == "conditional":
            return self._build_conditional_expr(operation, available_columns)
        elif op_type == "fill":
            return self._build_fill_expr(operation, available_columns)
        elif op_type == "round":
            return self._build_round_expr(operation, available_columns)
        elif op_type == "abs":
            return self._build_abs_expr(operation, available_columns)
        elif op_type == "clip":
            return self._build_clip_expr(operation, available_columns)
        else:
            raise TransformerConfigError(f"不支持的运算类型: {op_type}")

    def _build_arithmetic_expr(
        self,
        operation: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建算术运算表达式"""
        column = operation.get("column")
        operator = operation.get("operator")
        operand = operation.get("operand")

        if not column:
            raise TransformerConfigError("算术运算缺少 'column' 字段")
        if not operator:
            raise TransformerConfigError("算术运算缺少 'operator' 字段")
        if operand is None:
            raise TransformerConfigError("算术运算缺少 'operand' 字段")

        if column not in available_columns:
            raise InvalidColumnError(column, list(available_columns))

        if operator not in self.ARITHMETIC_OPERATORS:
            raise TransformerConfigError(f"不支持的算术操作符: {operator}")

        col_expr = pl.col(column)

        # operand 可以是列名或常量
        if isinstance(operand, str) and operand in available_columns:
            operand_expr = pl.col(operand)
        else:
            operand_expr = pl.lit(operand)

        if operator == "+":
            return col_expr + operand_expr
        elif operator == "-":
            return col_expr - operand_expr
        elif operator == "*":
            return col_expr * operand_expr
        elif operator == "/":
            return col_expr / operand_expr
        elif operator == "//":
            return col_expr // operand_expr
        elif operator == "%":
            return col_expr % operand_expr
        else:
            raise TransformerConfigError(f"不支持的算术操作符: {operator}")

    def _build_string_expr(
        self,
        operation: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建字符串运算表达式"""
        column = operation.get("column")
        str_op = operation.get("operation")
        params = operation.get("params", {})

        if not column:
            raise TransformerConfigError("字符串运算缺少 'column' 字段")
        if not str_op:
            raise TransformerConfigError("字符串运算缺少 'operation' 字段")

        if column not in available_columns:
            raise InvalidColumnError(column, list(available_columns))

        if str_op not in self.STRING_OPERATIONS:
            raise TransformerConfigError(f"不支持的字符串操作: {str_op}")

        col_expr = pl.col(column)

        if str_op == "upper":
            return col_expr.str.to_uppercase()
        elif str_op == "lower":
            return col_expr.str.to_lowercase()
        elif str_op == "strip":
            return col_expr.str.strip_chars()
        elif str_op == "lstrip":
            return col_expr.str.strip_chars_start()
        elif str_op == "rstrip":
            return col_expr.str.strip_chars_end()
        elif str_op == "replace":
            pattern = params.get("pattern", "")
            replacement = params.get("replacement", "")
            return col_expr.str.replace_all(pattern, replacement)
        elif str_op == "concat":
            suffix = params.get("suffix", "")
            return col_expr + pl.lit(suffix)
        elif str_op == "slice":
            offset = params.get("offset", 0)
            length = params.get("length")
            return col_expr.str.slice(offset, length)
        elif str_op == "length":
            return col_expr.str.len_chars()
        elif str_op == "reverse":
            return col_expr.str.reverse()
        elif str_op == "capitalize":
            return col_expr.str.to_titlecase()
        else:
            raise TransformerConfigError(f"不支持的字符串操作: {str_op}")

    def _build_datetime_expr(
        self,
        operation: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建日期运算表达式"""
        column = operation.get("column")
        dt_op = operation.get("operation")
        params = operation.get("params", {})

        if not column:
            raise TransformerConfigError("日期运算缺少 'column' 字段")
        if not dt_op:
            raise TransformerConfigError("日期运算缺少 'operation' 字段")

        if column not in available_columns:
            raise InvalidColumnError(column, list(available_columns))

        if dt_op not in self.DATETIME_OPERATIONS:
            raise TransformerConfigError(f"不支持的日期操作: {dt_op}")

        col_expr = pl.col(column)

        if dt_op == "extract_year":
            return col_expr.dt.year()
        elif dt_op == "extract_month":
            return col_expr.dt.month()
        elif dt_op == "extract_day":
            return col_expr.dt.day()
        elif dt_op == "extract_hour":
            return col_expr.dt.hour()
        elif dt_op == "extract_minute":
            return col_expr.dt.minute()
        elif dt_op == "extract_second":
            return col_expr.dt.second()
        elif dt_op == "extract_weekday":
            return col_expr.dt.weekday()
        elif dt_op == "extract_week":
            return col_expr.dt.week()
        elif dt_op == "add_days":
            days = params.get("days", 0)
            return col_expr + pl.duration(days=days)
        elif dt_op == "add_months":
            months = params.get("months", 0)
            return col_expr.dt.offset_by(f"{months}mo")
        elif dt_op == "truncate_day":
            return col_expr.dt.truncate("1d")
        elif dt_op == "truncate_month":
            return col_expr.dt.truncate("1mo")
        else:
            raise TransformerConfigError(f"不支持的日期操作: {dt_op}")

    def _build_conditional_expr(
        self,
        operation: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建条件运算表达式"""
        when_cond = operation.get("when")
        then_val = operation.get("then")
        otherwise_val = operation.get("otherwise")

        if not when_cond:
            raise TransformerConfigError("条件运算缺少 'when' 字段")

        # 构建条件表达式
        condition_expr = self._build_condition_expr(when_cond, available_columns)

        # 构建 when/then/otherwise
        expr = pl.when(condition_expr).then(pl.lit(then_val))
        if otherwise_val is not None:
            expr = expr.otherwise(pl.lit(otherwise_val))
        else:
            expr = expr.otherwise(pl.lit(None))

        return expr

    def _build_condition_expr(
        self,
        condition: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建条件表达式"""
        column = condition.get("column")
        operator = condition.get("operator", "==")
        value = condition.get("value")

        if not column:
            raise TransformerConfigError("条件缺少 'column' 字段")

        if column not in available_columns:
            raise InvalidColumnError(column, list(available_columns))

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
        elif operator == "is_null":
            return col_expr.is_null()
        elif operator == "is_not_null":
            return col_expr.is_not_null()
        else:
            raise TransformerConfigError(f"条件不支持的操作符: {operator}")

    def _build_fill_expr(
        self,
        operation: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建填充运算表达式"""
        column = operation.get("column")
        fill_op = operation.get("operation", "fill_null")
        value = operation.get("value")

        if not column:
            raise TransformerConfigError("填充运算缺少 'column' 字段")

        if column not in available_columns:
            raise InvalidColumnError(column, list(available_columns))

        col_expr = pl.col(column)

        if fill_op == "fill_null":
            return col_expr.fill_null(value)
        elif fill_op == "fill_nan":
            return col_expr.fill_nan(value)
        elif fill_op == "forward_fill":
            return col_expr.forward_fill()
        elif fill_op == "backward_fill":
            return col_expr.backward_fill()
        else:
            raise TransformerConfigError(f"不支持的填充操作: {fill_op}")

    def _build_round_expr(
        self,
        operation: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建四舍五入表达式"""
        column = operation.get("column")
        decimals = operation.get("decimals", 0)

        if not column:
            raise TransformerConfigError("四舍五入运算缺少 'column' 字段")

        if column not in available_columns:
            raise InvalidColumnError(column, list(available_columns))

        return pl.col(column).round(decimals)

    def _build_abs_expr(
        self,
        operation: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建绝对值表达式"""
        column = operation.get("column")

        if not column:
            raise TransformerConfigError("绝对值运算缺少 'column' 字段")

        if column not in available_columns:
            raise InvalidColumnError(column, list(available_columns))

        return pl.col(column).abs()

    def _build_clip_expr(
        self,
        operation: Dict[str, Any],
        available_columns: Set[str],
    ) -> pl.Expr:
        """构建裁剪表达式"""
        column = operation.get("column")
        lower = operation.get("lower")
        upper = operation.get("upper")

        if not column:
            raise TransformerConfigError("裁剪运算缺少 'column' 字段")

        if column not in available_columns:
            raise InvalidColumnError(column, list(available_columns))

        return pl.col(column).clip(lower, upper)
