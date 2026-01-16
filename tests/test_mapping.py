"""
字段映射转换器测试
"""

import pytest
import polars as pl

from qdata_transformer import PolarsFieldMappingTransformer
from qdata_transformer.exceptions import (
    MappingConfigError,
    InvalidColumnError,
    TransformExecutionError,
)


class TestPolarsFieldMappingTransformer:
    """字段映射转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = PolarsFieldMappingTransformer()

    def test_simple_mapping(self, sample_orders_data: pl.DataFrame) -> None:
        """测试简单字段映射"""
        config = {
            "mappings": [
                {"source": "order_id", "target": "id"},
                {"source": "customer_id", "target": "customer"},
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "id" in result.data.columns
        assert "customer" in result.data.columns
        assert result.data["id"].to_list() == sample_orders_data["order_id"].to_list()

    def test_expression_mapping(self, sample_orders_data: pl.DataFrame) -> None:
        """测试表达式映射"""
        config = {
            "mappings": [
                {"source": "order_id", "target": "id"},
                {
                    "source": ["quantity", "price"],
                    "target": "amount",
                    "transform": "expression",
                    "params": {"expr": "quantity * price"},
                },
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "amount" in result.data.columns
        expected_amounts = [
            q * p
            for q, p in zip(
                sample_orders_data["quantity"].to_list(),
                sample_orders_data["price"].to_list(),
            )
        ]
        assert result.data["amount"].to_list() == expected_amounts

    def test_constant_mapping(self, sample_orders_data: pl.DataFrame) -> None:
        """测试常量映射"""
        config = {
            "mappings": [
                {"source": "order_id", "target": "id"},
                {"target": "source", "transform": "constant", "params": {"value": "web"}},
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "source" in result.data.columns
        assert all(v == "web" for v in result.data["source"].to_list())

    def test_cast_mapping(self, sample_orders_data: pl.DataFrame) -> None:
        """测试类型转换映射"""
        config = {
            "mappings": [
                {"source": "quantity", "target": "qty_str", "transform": "cast", "params": {"dtype": "str"}},
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert result.data["qty_str"].dtype == pl.Utf8

    def test_drop_unmapped_false(self, sample_orders_data: pl.DataFrame) -> None:
        """测试保留未映射的列"""
        config = {
            "mappings": [
                {"source": "order_id", "target": "id"},
            ],
            "drop_unmapped": False,
        }

        result = self.transformer.execute(sample_orders_data, config)

        # 应该保留原始列
        assert "customer_id" in result.data.columns
        assert "quantity" in result.data.columns

    def test_missing_mappings_config(self, sample_orders_data: pl.DataFrame) -> None:
        """测试缺少 mappings 配置"""
        with pytest.raises(MappingConfigError):
            self.transformer.execute(sample_orders_data, {})

    def test_invalid_source_column(self, sample_orders_data: pl.DataFrame) -> None:
        """测试无效的源列"""
        config = {
            "mappings": [
                {"source": "nonexistent_column", "target": "id"},
            ]
        }

        with pytest.raises(TransformExecutionError) as exc_info:
            self.transformer.execute(sample_orders_data, config)
        # 验证原始异常是 InvalidColumnError
        assert isinstance(exc_info.value.original_error, InvalidColumnError)

    def test_constant_without_value(self, sample_orders_data: pl.DataFrame) -> None:
        """测试常量映射缺少 value"""
        config = {
            "mappings": [
                {"target": "status", "transform": "constant", "params": {}},
            ]
        }

        with pytest.raises(MappingConfigError):
            self.transformer.execute(sample_orders_data, config)

    def test_expression_without_expr(self, sample_orders_data: pl.DataFrame) -> None:
        """测试表达式映射缺少 expr"""
        config = {
            "mappings": [
                {"source": ["a", "b"], "target": "c", "transform": "expression", "params": {}},
            ]
        }

        with pytest.raises(MappingConfigError):
            self.transformer.execute(sample_orders_data, config)
