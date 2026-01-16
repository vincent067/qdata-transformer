"""
性能和边界测试

测试转换器的性能和边界情况。
"""

import pytest
import polars as pl
import time
from typing import List, Any

from qdata_transformer import (
    TransformChain,
    PolarsFieldMappingTransformer,
    PolarsFilterTransformer,
    DuckDBAggregationTransformer,
)


class TestPerformance:
    """性能测试类"""

    @pytest.fixture
    def large_dataset(self) -> pl.DataFrame:
        """生成大数据集"""
        import random
        n = 100000  # 10万行

        return pl.DataFrame({
            "id": list(range(n)),
            "customer_id": [f"C{random.randint(1, 1000):04d}" for _ in range(n)],
            "product_id": [f"P{random.randint(1, 100):03d}" for _ in range(n)],
            "quantity": [random.randint(1, 100) for _ in range(n)],
            "price": [round(random.uniform(10, 1000), 2) for _ in range(n)],
            "status": [random.choice(["pending", "completed", "cancelled"]) for _ in range(n)],
        })

    @pytest.mark.slow
    def test_mapping_performance(self, large_dataset: pl.DataFrame) -> None:
        """测试映射转换器性能"""
        transformer = PolarsFieldMappingTransformer()
        config = {
            "mappings": [
                {"source": "id", "target": "order_id"},
                {"source": "customer_id", "target": "customer"},
                {"source": ["quantity", "price"], "target": "amount",
                 "transform": "expression", "params": {"expr": "quantity * price"}},
                {"target": "source", "transform": "constant", "params": {"value": "test"}}
            ]
        }

        start = time.time()
        result = transformer.execute(large_dataset, config)
        elapsed = time.time() - start

        assert len(result.data) == len(large_dataset)
        # 10万行应该在 1 秒内完成
        assert elapsed < 1.0, f"性能测试失败: 耗时 {elapsed:.3f}s > 1.0s"

    @pytest.mark.slow
    def test_filter_performance(self, large_dataset: pl.DataFrame) -> None:
        """测试过滤转换器性能"""
        transformer = PolarsFilterTransformer()
        config = {
            "conditions": [
                {"column": "quantity", "operator": ">", "value": 50},
                {"column": "status", "operator": "==", "value": "completed"}
            ],
            "logic": "and"
        }

        start = time.time()
        result = transformer.execute(large_dataset, config)
        elapsed = time.time() - start

        # 10万行应该在 0.5 秒内完成
        assert elapsed < 0.5, f"性能测试失败: 耗时 {elapsed:.3f}s > 0.5s"

    @pytest.mark.slow
    def test_aggregation_performance(self, large_dataset: pl.DataFrame) -> None:
        """测试聚合转换器性能"""
        transformer = DuckDBAggregationTransformer()
        config = {
            "group_by": ["customer_id"],
            "aggregations": [
                {"field": "quantity", "function": "sum", "alias": "total_quantity"},
                {"field": "price", "function": "avg", "alias": "avg_price"},
                {"field": "id", "function": "count", "alias": "order_count"}
            ]
        }

        start = time.time()
        result = transformer.execute(large_dataset, config)
        elapsed = time.time() - start

        # 10万行聚合应该在 1 秒内完成
        assert elapsed < 1.0, f"性能测试失败: 耗时 {elapsed:.3f}s > 1.0s"

    @pytest.mark.slow
    def test_chain_performance(self, large_dataset: pl.DataFrame) -> None:
        """测试转换链性能"""
        chain = (
            TransformChain()
            .add("polars_filter", {
                "conditions": [{"column": "quantity", "operator": ">", "value": 10}]
            })
            .add("polars_field_mapping", {
                "mappings": [
                    {"source": "id", "target": "order_id"},
                    {"source": "customer_id", "target": "customer"},
                    {"source": ["quantity", "price"], "target": "amount",
                     "transform": "expression", "params": {"expr": "quantity * price"}}
                ]
            })
            .add("duckdb_aggregation", {
                "group_by": ["customer"],
                "aggregations": [
                    {"field": "amount", "function": "sum", "alias": "total_amount"},
                    {"field": "order_id", "function": "count", "alias": "order_count"}
                ]
            })
        )

        start = time.time()
        result = chain.execute(large_dataset)
        elapsed = time.time() - start

        # 完整流水线应该在 2 秒内完成
        assert elapsed < 2.0, f"性能测试失败: 耗时 {elapsed:.3f}s > 2.0s"


class TestEdgeCases:
    """边界测试类"""

    def test_empty_dataframe(self) -> None:
        """测试空数据集"""
        data = pl.DataFrame({"a": [], "b": []})
        transformer = PolarsFieldMappingTransformer()

        config = {
            "mappings": [{"source": "a", "target": "c"}]
        }

        result = transformer.execute(data, config)

        assert len(result.data) == 0
        assert "c" in result.data.columns

    def test_single_row(self) -> None:
        """测试单行数据"""
        data = pl.DataFrame({"a": [1], "b": ["test"]})
        transformer = PolarsFieldMappingTransformer()

        config = {
            "mappings": [{"source": "a", "target": "c"}]
        }

        result = transformer.execute(data, config)

        assert len(result.data) == 1
        assert result.data["c"][0] == 1

    def test_null_values(self) -> None:
        """测试空值处理"""
        data = pl.DataFrame({
            "a": [1, None, 3],
            "b": ["x", None, "z"]
        })
        transformer = PolarsFieldMappingTransformer()

        config = {
            "mappings": [
                {"source": "a", "target": "c"},
                {"source": "b", "target": "d"}
            ]
        }

        result = transformer.execute(data, config)

        assert result.data["c"].null_count() == 1
        assert result.data["d"].null_count() == 1

    def test_large_string_values(self) -> None:
        """测试大字符串值"""
        large_string = "x" * 10000  # 1万字符
        data = pl.DataFrame({
            "text": [large_string, large_string]
        })
        transformer = PolarsFieldMappingTransformer()

        config = {
            "mappings": [{"source": "text", "target": "content"}]
        }

        result = transformer.execute(data, config)

        assert result.data["content"][0] == large_string

    def test_special_characters_in_column_names(self) -> None:
        """测试列名中的特殊字符"""
        data = pl.DataFrame({
            "column with space": [1, 2],
            "column-with-dash": [3, 4]
        })
        transformer = PolarsFieldMappingTransformer()

        config = {
            "mappings": [
                {"source": "column with space", "target": "new_col1"},
                {"source": "column-with-dash", "target": "new_col2"}
            ]
        }

        result = transformer.execute(data, config)

        assert "new_col1" in result.data.columns
        assert "new_col2" in result.data.columns

    def test_unicode_values(self) -> None:
        """测试 Unicode 值"""
        data = pl.DataFrame({
            "name": ["Alice", "中文名字", "🎉emoji"],
            "value": [1, 2, 3]
        })
        transformer = PolarsFieldMappingTransformer()

        config = {
            "mappings": [
                {"source": "name", "target": "username"},
                {"source": "value", "target": "val"}
            ]
        }

        result = transformer.execute(data, config)

        assert result.data["username"].to_list() == ["Alice", "中文名字", "🎉emoji"]

    def test_very_large_numbers(self) -> None:
        """测试超大数值"""
        data = pl.DataFrame({
            "big_int": [10**18, 10**17, 10**16],
            "big_float": [1e308, 1e-308, 0.0]
        })
        transformer = PolarsFieldMappingTransformer()

        config = {
            "mappings": [
                {"source": "big_int", "target": "int_col"},
                {"source": "big_float", "target": "float_col"}
            ]
        }

        result = transformer.execute(data, config)

        assert result.data["int_col"][0] == 10**18
        assert result.data["float_col"][0] == 1e308

    def test_aggregation_empty_groups(self) -> None:
        """测试聚合空分组"""
        data = pl.DataFrame({
            "category": ["A", "A", "B"],
            "value": [1, 2, 3]
        })
        transformer = DuckDBAggregationTransformer()

        config = {
            "group_by": ["category"],
            "aggregations": [
                {"field": "value", "function": "sum", "alias": "total"}
            ],
            "having": "total > 100"  # 没有分组满足条件
        }

        result = transformer.execute(data, config)

        assert len(result.data) == 0

    def test_filter_no_match(self) -> None:
        """测试过滤无匹配"""
        data = pl.DataFrame({
            "value": [1, 2, 3, 4, 5]
        })
        transformer = PolarsFilterTransformer()

        config = {
            "conditions": [
                {"column": "value", "operator": ">", "value": 100}
            ]
        }

        result = transformer.execute(data, config)

        assert len(result.data) == 0

    def test_filter_all_match(self) -> None:
        """测试过滤全匹配"""
        data = pl.DataFrame({
            "value": [10, 20, 30, 40, 50]
        })
        transformer = PolarsFilterTransformer()

        config = {
            "conditions": [
                {"column": "value", "operator": ">", "value": 0}
            ]
        }

        result = transformer.execute(data, config)

        assert len(result.data) == 5

    def test_chain_empty_steps(self) -> None:
        """测试空转换链"""
        chain = TransformChain()
        data = pl.DataFrame({"a": [1, 2, 3]})

        result = chain.execute(data)

        assert len(result.data) == 3
        assert result.metadata["chain_steps"] == 0

    def test_chain_all_disabled_steps(self) -> None:
        """测试全部禁用的步骤"""
        chain = (
            TransformChain()
            .add("polars_field_mapping", {"mappings": []}, enabled=False)
            .add("polars_filter", {"conditions": []}, enabled=False)
        )
        data = pl.DataFrame({"a": [1, 2, 3]})

        result = chain.execute(data)

        assert len(result.data) == 3
