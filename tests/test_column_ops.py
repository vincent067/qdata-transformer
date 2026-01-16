"""
列运算转换器测试
"""

import pytest
import polars as pl
from datetime import datetime

from qdata_transformer import PolarsColumnOpsTransformer
from qdata_transformer.exceptions import TransformerConfigError, TransformExecutionError


class TestPolarsColumnOpsTransformer:
    """列运算转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = PolarsColumnOpsTransformer()

    def test_arithmetic_add(self, sample_orders_data: pl.DataFrame) -> None:
        """测试加法运算"""
        config = {
            "operations": [
                {"type": "arithmetic", "column": "quantity", "operator": "+", "operand": 10, "result": "qty_plus_10"}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "qty_plus_10" in result.data.columns
        expected = [q + 10 for q in sample_orders_data["quantity"].to_list()]
        assert result.data["qty_plus_10"].to_list() == expected

    def test_arithmetic_multiply_columns(self, sample_orders_data: pl.DataFrame) -> None:
        """测试列间乘法运算"""
        config = {
            "operations": [
                {"type": "arithmetic", "column": "quantity", "operator": "*", "operand": "price", "result": "amount"}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "amount" in result.data.columns
        expected = [
            q * p for q, p in zip(
                sample_orders_data["quantity"].to_list(),
                sample_orders_data["price"].to_list()
            )
        ]
        assert result.data["amount"].to_list() == expected

    def test_string_upper(self, sample_orders_data: pl.DataFrame) -> None:
        """测试字符串大写"""
        config = {
            "operations": [
                {"type": "string", "column": "status", "operation": "upper", "result": "status_upper"}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "status_upper" in result.data.columns
        for i, status in enumerate(result.data["status_upper"].to_list()):
            assert status == sample_orders_data["status"][i].upper()

    def test_string_lower(self, sample_orders_data: pl.DataFrame) -> None:
        """测试字符串小写"""
        config = {
            "operations": [
                {"type": "string", "column": "order_id", "operation": "lower", "result": "order_id_lower"}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "order_id_lower" in result.data.columns
        for val in result.data["order_id_lower"].to_list():
            assert val.islower() or val.isdigit()

    def test_string_replace(self, sample_orders_data: pl.DataFrame) -> None:
        """测试字符串替换"""
        config = {
            "operations": [
                {
                    "type": "string",
                    "column": "order_id",
                    "operation": "replace",
                    "params": {"pattern": "O", "replacement": "ORDER-"},
                    "result": "order_id_new"
                }
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "order_id_new" in result.data.columns
        for val in result.data["order_id_new"].to_list():
            assert "ORDER-" in val

    def test_conditional(self, sample_orders_data: pl.DataFrame) -> None:
        """测试条件运算"""
        config = {
            "operations": [
                {
                    "type": "conditional",
                    "when": {"column": "quantity", "operator": ">=", "value": 3},
                    "then": "high",
                    "otherwise": "low",
                    "result": "quantity_level"
                }
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "quantity_level" in result.data.columns
        for i, level in enumerate(result.data["quantity_level"].to_list()):
            if sample_orders_data["quantity"][i] >= 3:
                assert level == "high"
            else:
                assert level == "low"

    def test_round(self) -> None:
        """测试四舍五入"""
        data = pl.DataFrame({
            "value": [1.234, 5.678, 9.999]
        })

        config = {
            "operations": [
                {"type": "round", "column": "value", "decimals": 2, "result": "value_rounded"}
            ]
        }

        result = self.transformer.execute(data, config)

        assert result.data["value_rounded"].to_list() == [1.23, 5.68, 10.0]

    def test_abs(self) -> None:
        """测试绝对值"""
        data = pl.DataFrame({
            "value": [-10, 20, -30, 40]
        })

        config = {
            "operations": [
                {"type": "abs", "column": "value", "result": "value_abs"}
            ]
        }

        result = self.transformer.execute(data, config)

        assert result.data["value_abs"].to_list() == [10, 20, 30, 40]

    def test_clip(self) -> None:
        """测试裁剪"""
        data = pl.DataFrame({
            "value": [1, 5, 10, 15, 20]
        })

        config = {
            "operations": [
                {"type": "clip", "column": "value", "lower": 5, "upper": 15, "result": "value_clipped"}
            ]
        }

        result = self.transformer.execute(data, config)

        assert result.data["value_clipped"].to_list() == [5, 5, 10, 15, 15]

    def test_fill_null(self) -> None:
        """测试空值填充"""
        data = pl.DataFrame({
            "value": [1, None, 3, None, 5]
        })

        config = {
            "operations": [
                {"type": "fill", "column": "value", "operation": "fill_null", "value": 0, "result": "value_filled"}
            ]
        }

        result = self.transformer.execute(data, config)

        assert result.data["value_filled"].to_list() == [1, 0, 3, 0, 5]

    def test_multiple_operations(self, sample_orders_data: pl.DataFrame) -> None:
        """测试多个运算"""
        config = {
            "operations": [
                {"type": "arithmetic", "column": "quantity", "operator": "*", "operand": "price", "result": "amount"},
                {"type": "string", "column": "status", "operation": "upper", "result": "status_upper"},
                {"type": "round", "column": "price", "decimals": 0, "result": "price_rounded"}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "amount" in result.data.columns
        assert "status_upper" in result.data.columns
        assert "price_rounded" in result.data.columns

    def test_missing_operations_config(self, sample_orders_data: pl.DataFrame) -> None:
        """测试缺少 operations 配置"""
        with pytest.raises(TransformerConfigError):
            self.transformer.execute(sample_orders_data, {})

    def test_unsupported_operation_type(self, sample_orders_data: pl.DataFrame) -> None:
        """测试不支持的运算类型"""
        config = {
            "operations": [
                {"type": "unsupported", "column": "quantity", "result": "new_col"}
            ]
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(sample_orders_data, config)

    def test_invalid_column(self, sample_orders_data: pl.DataFrame) -> None:
        """测试无效列名"""
        config = {
            "operations": [
                {"type": "arithmetic", "column": "nonexistent", "operator": "+", "operand": 1, "result": "new_col"}
            ]
        }

        with pytest.raises(TransformExecutionError):
            self.transformer.execute(sample_orders_data, config)
