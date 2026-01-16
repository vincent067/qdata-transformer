"""
聚合转换器测试
"""

import pytest
import polars as pl

from qdata_transformer import DuckDBAggregationTransformer, DuckDBSQLTransformer
from qdata_transformer.exceptions import AggregationConfigError


class TestDuckDBAggregationTransformer:
    """聚合转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = DuckDBAggregationTransformer()

    def test_simple_aggregation(self, sample_orders_data: pl.DataFrame) -> None:
        """测试简单聚合"""
        config = {
            "group_by": ["customer_id"],
            "aggregations": [
                {"field": "quantity", "function": "sum", "alias": "total_quantity"},
                {"field": "order_id", "function": "count", "alias": "order_count"},
            ],
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "customer_id" in result.data.columns
        assert "total_quantity" in result.data.columns
        assert "order_count" in result.data.columns

    def test_global_aggregation(self, sample_orders_data: pl.DataFrame) -> None:
        """测试全局聚合（无分组）"""
        config = {
            "aggregations": [
                {"field": "quantity", "function": "sum", "alias": "total_quantity"},
                {"field": "price", "function": "avg", "alias": "avg_price"},
            ],
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert len(result.data) == 1
        assert "total_quantity" in result.data.columns
        assert "avg_price" in result.data.columns

    def test_aggregation_with_having(self, sample_orders_data: pl.DataFrame) -> None:
        """测试带 HAVING 条件的聚合"""
        config = {
            "group_by": ["customer_id"],
            "aggregations": [
                {"field": "quantity", "function": "sum", "alias": "total_quantity"},
            ],
            "having": "total_quantity > 2",
        }

        result = self.transformer.execute(sample_orders_data, config)

        # 过滤后应该只有 total_quantity > 2 的记录
        for qty in result.data["total_quantity"].to_list():
            assert qty > 2

    def test_aggregation_with_order_by(self, sample_orders_data: pl.DataFrame) -> None:
        """测试带排序的聚合"""
        config = {
            "group_by": ["customer_id"],
            "aggregations": [
                {"field": "quantity", "function": "sum", "alias": "total_quantity"},
            ],
            "order_by": ["total_quantity DESC"],
        }

        result = self.transformer.execute(sample_orders_data, config)

        # 验证降序排列
        quantities = result.data["total_quantity"].to_list()
        assert quantities == sorted(quantities, reverse=True)

    def test_missing_aggregations_config(self, sample_orders_data: pl.DataFrame) -> None:
        """测试缺少 aggregations 配置"""
        config = {"group_by": ["customer_id"]}

        with pytest.raises(AggregationConfigError):
            self.transformer.execute(sample_orders_data, config)

    def test_unsupported_function(self, sample_orders_data: pl.DataFrame) -> None:
        """测试不支持的聚合函数"""
        config = {
            "aggregations": [
                {"field": "quantity", "function": "unsupported_func", "alias": "result"},
            ],
        }

        with pytest.raises(AggregationConfigError):
            self.transformer.execute(sample_orders_data, config)


class TestDuckDBSQLTransformer:
    """SQL 转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = DuckDBSQLTransformer()

    def test_simple_sql(self, sample_orders_data: pl.DataFrame) -> None:
        """测试简单 SQL 查询"""
        config = {
            "sql": "SELECT order_id, customer_id, quantity * price as amount FROM data"
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "order_id" in result.data.columns
        assert "customer_id" in result.data.columns
        assert "amount" in result.data.columns

    def test_sql_with_where(self, sample_orders_data: pl.DataFrame) -> None:
        """测试带 WHERE 条件的 SQL"""
        config = {
            "sql": "SELECT * FROM data WHERE status = 'completed'"
        }

        result = self.transformer.execute(sample_orders_data, config)

        # 所有结果的 status 应该都是 completed
        for status in result.data["status"].to_list():
            assert status == "completed"

    def test_sql_aggregation(self, sample_orders_data: pl.DataFrame) -> None:
        """测试 SQL 聚合查询"""
        config = {
            "sql": """
                SELECT 
                    customer_id,
                    COUNT(*) as order_count,
                    SUM(quantity) as total_quantity
                FROM data
                GROUP BY customer_id
                ORDER BY total_quantity DESC
            """
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "customer_id" in result.data.columns
        assert "order_count" in result.data.columns
        assert "total_quantity" in result.data.columns

    def test_missing_sql_config(self, sample_orders_data: pl.DataFrame) -> None:
        """测试缺少 sql 配置"""
        with pytest.raises(AggregationConfigError):
            self.transformer.execute(sample_orders_data, {})

    def test_empty_sql(self, sample_orders_data: pl.DataFrame) -> None:
        """测试空 SQL"""
        config = {"sql": "   "}

        with pytest.raises(AggregationConfigError):
            self.transformer.execute(sample_orders_data, config)
