"""
批量映射转换器测试
"""

import pytest
import polars as pl

from qdata_transformer import PolarsMultiMappingTransformer
from qdata_transformer.exceptions import MappingConfigError, TransformExecutionError


class TestPolarsMultiMappingTransformer:
    """批量映射转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = PolarsMultiMappingTransformer()

    def test_simple_mapping(self, sample_orders_data: pl.DataFrame) -> None:
        """测试简单映射"""
        config = {
            "mappings": [
                {"source": "order_id", "target": "id"},
                {"source": "customer_id", "target": "customer"}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "id" in result.data.columns
        assert "customer" in result.data.columns

    def test_constant_mapping(self, sample_orders_data: pl.DataFrame) -> None:
        """测试常量映射"""
        config = {
            "mappings": [
                {"source": "order_id", "target": "id"},
                {"target": "source", "transform": "constant", "params": {"value": "api"}}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert all(v == "api" for v in result.data["source"].to_list())

    def test_nested_field(self, sample_nested_data: pl.DataFrame) -> None:
        """测试嵌套字段访问"""
        config = {
            "mappings": [
                {"source": "id", "target": "id"},
                {"source": "customer.name", "target": "customer_name", "transform": "nested"}
            ]
        }

        result = self.transformer.execute(sample_nested_data, config)

        assert "customer_name" in result.data.columns
        assert result.data["customer_name"].to_list() == ["Alice", "Bob", "Charlie"]

    def test_coalesce_mapping(self) -> None:
        """测试 coalesce 映射"""
        data = pl.DataFrame({
            "a": [1, None, 3],
            "b": [None, 2, None],
            "c": [10, 20, 30]
        })

        config = {
            "mappings": [
                {"source": ["a", "b", "c"], "target": "result", "transform": "coalesce"}
            ]
        }

        result = self.transformer.execute(data, config)

        assert result.data["result"].to_list() == [1, 2, 3]

    def test_concat_mapping(self) -> None:
        """测试字符串连接映射"""
        data = pl.DataFrame({
            "first_name": ["Alice", "Bob"],
            "last_name": ["Smith", "Jones"]
        })

        config = {
            "mappings": [
                {
                    "source": ["first_name", "last_name"],
                    "target": "full_name",
                    "transform": "concat",
                    "params": {"separator": " "}
                }
            ]
        }

        result = self.transformer.execute(data, config)

        assert result.data["full_name"].to_list() == ["Alice Smith", "Bob Jones"]

    def test_cast_mapping(self, sample_orders_data: pl.DataFrame) -> None:
        """测试类型转换映射"""
        config = {
            "mappings": [
                {"source": "quantity", "target": "qty_str", "transform": "cast", "params": {"dtype": "str"}}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert result.data["qty_str"].dtype == pl.Utf8

    def test_explode_mapping(self) -> None:
        """测试数组展开映射"""
        data = pl.DataFrame({
            "id": [1, 2],
            "items": [[1, 2, 3], [4, 5]]
        })

        config = {
            "mappings": [
                {"source": "id", "target": "id"},
                {"source": "items", "target": "item", "transform": "explode"}
            ]
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 5  # 3 + 2 = 5

    def test_with_filter(self, sample_orders_data: pl.DataFrame) -> None:
        """测试带过滤的映射"""
        config = {
            "mappings": [
                {"source": "order_id", "target": "id"},
                {"source": "quantity", "target": "qty"}
            ],
            "filter": {
                "condition": "qty > 2"  # 使用映射后的列名
            }
        }

        result = self.transformer.execute(sample_orders_data, config)

        for qty in result.data["qty"].to_list():
            assert qty > 2

    def test_drop_unmapped_false(self, sample_orders_data: pl.DataFrame) -> None:
        """测试保留未映射列"""
        config = {
            "mappings": [
                {"source": "order_id", "target": "id"}
            ],
            "drop_unmapped": False
        }

        result = self.transformer.execute(sample_orders_data, config)

        # 原始列应该保留
        assert "customer_id" in result.data.columns

    def test_missing_mappings_config(self, sample_orders_data: pl.DataFrame) -> None:
        """测试缺少 mappings 配置"""
        with pytest.raises(MappingConfigError):
            self.transformer.execute(sample_orders_data, {})

    def test_invalid_transform_type(self, sample_orders_data: pl.DataFrame) -> None:
        """测试无效的转换类型"""
        config = {
            "mappings": [
                {"source": "order_id", "target": "id", "transform": "invalid_type"}
            ]
        }

        with pytest.raises(MappingConfigError):
            self.transformer.execute(sample_orders_data, config)

    def test_invalid_source_column(self, sample_orders_data: pl.DataFrame) -> None:
        """测试无效的源列"""
        config = {
            "mappings": [
                {"source": "nonexistent", "target": "id"}
            ]
        }

        with pytest.raises(TransformExecutionError):
            self.transformer.execute(sample_orders_data, config)

    def test_constant_without_source(self, sample_orders_data: pl.DataFrame) -> None:
        """测试常量映射不需要源列"""
        config = {
            "mappings": [
                {"target": "const_col", "transform": "constant", "params": {"value": 42}}
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert all(v == 42 for v in result.data["const_col"].to_list())

    def test_filter_with_operators(self, sample_orders_data: pl.DataFrame) -> None:
        """测试不同过滤操作符"""
        # 测试 == (使用 drop_unmapped: False 保留原始列)
        config1 = {
            "mappings": [{"source": "status", "target": "status"}],
            "filter": {"condition": "status == 'completed'"}
        }
        result1 = self.transformer.execute(sample_orders_data, config1)
        for s in result1.data["status"].to_list():
            assert s == "completed"

        # 测试 !=
        config2 = {
            "mappings": [{"source": "status", "target": "status"}],
            "filter": {"condition": "status != 'cancelled'"}
        }
        result2 = self.transformer.execute(sample_orders_data, config2)
        for s in result2.data["status"].to_list():
            assert s != "cancelled"

        # 测试 <
        config3 = {
            "mappings": [{"source": "quantity", "target": "qty"}],
            "filter": {"condition": "qty < 3"}
        }
        result3 = self.transformer.execute(sample_orders_data, config3)
        for q in result3.data["qty"].to_list():
            assert q < 3

        # 测试 >=
        config4 = {
            "mappings": [{"source": "quantity", "target": "qty"}],
            "filter": {"condition": "qty >= 2"}
        }
        result4 = self.transformer.execute(sample_orders_data, config4)
        for q in result4.data["qty"].to_list():
            assert q >= 2

        # 测试 <=
        config5 = {
            "mappings": [{"source": "quantity", "target": "qty"}],
            "filter": {"condition": "qty <= 3"}
        }
        result5 = self.transformer.execute(sample_orders_data, config5)
        for q in result5.data["qty"].to_list():
            assert q <= 3
