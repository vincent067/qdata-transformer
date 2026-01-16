"""
数据过滤转换器测试
"""

import pytest
import polars as pl

from qdata_transformer import PolarsFilterTransformer
from qdata_transformer.exceptions import TransformerConfigError, TransformExecutionError


class TestPolarsFilterTransformer:
    """过滤转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = PolarsFilterTransformer()

    def test_simple_filter(self, sample_orders_data: pl.DataFrame) -> None:
        """测试简单过滤"""
        config = {
            "conditions": [
                {"column": "quantity", "operator": ">", "value": 2}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        # 所有结果的 quantity 应该 > 2
        for qty in result.data["quantity"].to_list():
            assert qty > 2

    def test_equality_filter(self, sample_orders_data: pl.DataFrame) -> None:
        """测试相等过滤"""
        config = {
            "conditions": [
                {"column": "status", "operator": "==", "value": "completed"}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        for status in result.data["status"].to_list():
            assert status == "completed"

    def test_in_filter(self, sample_orders_data: pl.DataFrame) -> None:
        """测试 IN 过滤"""
        config = {
            "conditions": [
                {"column": "customer_id", "operator": "in", "value": ["C001", "C002"]}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        for cid in result.data["customer_id"].to_list():
            assert cid in ["C001", "C002"]

    def test_not_in_filter(self, sample_orders_data: pl.DataFrame) -> None:
        """测试 NOT IN 过滤"""
        config = {
            "conditions": [
                {"column": "status", "operator": "not_in", "value": ["cancelled", "pending"]}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        for status in result.data["status"].to_list():
            assert status not in ["cancelled", "pending"]

    def test_between_filter(self, sample_orders_data: pl.DataFrame) -> None:
        """测试范围过滤"""
        config = {
            "conditions": [
                {"column": "quantity", "operator": "between", "min": 2, "max": 3}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        for qty in result.data["quantity"].to_list():
            assert 2 <= qty <= 3

    def test_contains_filter(self, sample_orders_data: pl.DataFrame) -> None:
        """测试字符串包含过滤"""
        config = {
            "conditions": [
                {"column": "order_id", "operator": "contains", "value": "001"}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        for oid in result.data["order_id"].to_list():
            assert "001" in oid

    def test_multiple_conditions_and(self, sample_orders_data: pl.DataFrame) -> None:
        """测试多条件 AND"""
        config = {
            "conditions": [
                {"column": "quantity", "operator": ">=", "value": 2},
                {"column": "status", "operator": "==", "value": "completed"}
            ],
            "logic": "and"
        }

        result = self.transformer.execute(sample_orders_data, config)

        for row in result.data.iter_rows(named=True):
            assert row["quantity"] >= 2
            assert row["status"] == "completed"

    def test_multiple_conditions_or(self, sample_orders_data: pl.DataFrame) -> None:
        """测试多条件 OR"""
        config = {
            "conditions": [
                {"column": "quantity", "operator": ">=", "value": 5},
                {"column": "status", "operator": "==", "value": "pending"}
            ],
            "logic": "or"
        }

        result = self.transformer.execute(sample_orders_data, config)

        for row in result.data.iter_rows(named=True):
            assert row["quantity"] >= 5 or row["status"] == "pending"

    def test_drop_null(self) -> None:
        """测试空值过滤"""
        data = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "value": [10, None, 30, None]
        })

        config = {
            "drop_null": ["value"]
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 2
        assert result.data["value"].null_count() == 0

    def test_is_null_filter(self) -> None:
        """测试 is_null 过滤"""
        data = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "value": [10, None, 30, None]
        })

        config = {
            "conditions": [
                {"column": "value", "operator": "is_null"}
            ]
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 2
        assert result.data["value"].null_count() == 2

    def test_invalid_operator(self, sample_orders_data: pl.DataFrame) -> None:
        """测试无效操作符"""
        config = {
            "conditions": [
                {"column": "quantity", "operator": "invalid_op", "value": 1}
            ]
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(sample_orders_data, config)

    def test_invalid_column(self, sample_orders_data: pl.DataFrame) -> None:
        """测试无效列名"""
        config = {
            "conditions": [
                {"column": "nonexistent", "operator": "==", "value": 1}
            ]
        }

        with pytest.raises(TransformExecutionError):
            self.transformer.execute(sample_orders_data, config)

    def test_between_missing_params(self, sample_orders_data: pl.DataFrame) -> None:
        """测试 between 缺少参数"""
        config = {
            "conditions": [
                {"column": "quantity", "operator": "between", "min": 1}  # 缺少 max
            ]
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(sample_orders_data, config)
