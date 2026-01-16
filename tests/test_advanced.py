"""
高级转换器测试（JOIN 和窗口函数）
"""

import pytest
import polars as pl

from qdata_transformer import DuckDBJoinTransformer, DuckDBWindowTransformer
from qdata_transformer.exceptions import TransformerConfigError, TransformExecutionError


class TestDuckDBJoinTransformer:
    """JOIN 转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = DuckDBJoinTransformer()

    def test_inner_join(self) -> None:
        """测试内连接"""
        left_data = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })

        right_data = [
            {"id": 1, "city": "Beijing"},
            {"id": 2, "city": "Shanghai"},
            {"id": 4, "city": "Guangzhou"}
        ]

        config = {
            "right_data": right_data,
            "join_type": "inner",
            "left_on": ["id"],
            "right_on": ["id"]
        }

        result = self.transformer.execute(left_data, config)

        assert len(result.data) == 2  # id=1, id=2
        assert "city_right" in result.data.columns

    def test_left_join(self) -> None:
        """测试左连接"""
        left_data = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })

        right_data = [
            {"id": 1, "city": "Beijing"},
            {"id": 2, "city": "Shanghai"}
        ]

        config = {
            "right_data": right_data,
            "join_type": "left",
            "left_on": ["id"],
            "right_on": ["id"]
        }

        result = self.transformer.execute(left_data, config)

        assert len(result.data) == 3  # 所有左表记录
        # id=3 的 city 应该为 null
        cities = result.data["city_right"].to_list()
        assert None in cities

    def test_cross_join(self) -> None:
        """测试交叉连接"""
        left_data = pl.DataFrame({
            "id": [1, 2],
            "name": ["A", "B"]
        })

        right_data = [
            {"color": "red"},
            {"color": "blue"}
        ]

        config = {
            "right_data": right_data,
            "join_type": "cross"
        }

        result = self.transformer.execute(left_data, config)

        assert len(result.data) == 4  # 2 * 2 = 4

    def test_custom_suffix(self) -> None:
        """测试自定义后缀"""
        left_data = pl.DataFrame({
            "id": [1, 2],
            "value": [10, 20]
        })

        right_data = [
            {"id": 1, "value": 100},
            {"id": 2, "value": 200}
        ]

        config = {
            "right_data": right_data,
            "join_type": "inner",
            "left_on": ["id"],
            "right_on": ["id"],
            "suffix": "_r"
        }

        result = self.transformer.execute(left_data, config)

        assert "value_r" in result.data.columns

    def test_multiple_join_keys(self) -> None:
        """测试多键连接"""
        left_data = pl.DataFrame({
            "year": [2024, 2024, 2025],
            "month": [1, 2, 1],
            "value": [100, 200, 300]
        })

        right_data = [
            {"year": 2024, "month": 1, "budget": 150},
            {"year": 2024, "month": 2, "budget": 250}
        ]

        config = {
            "right_data": right_data,
            "join_type": "inner",
            "left_on": ["year", "month"],
            "right_on": ["year", "month"]
        }

        result = self.transformer.execute(left_data, config)

        assert len(result.data) == 2

    def test_missing_right_data(self) -> None:
        """测试缺少右表数据"""
        left_data = pl.DataFrame({"id": [1, 2]})

        config = {
            "join_type": "inner",
            "left_on": ["id"],
            "right_on": ["id"]
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(left_data, config)

    def test_mismatched_join_keys(self) -> None:
        """测试不匹配的连接键"""
        left_data = pl.DataFrame({"id": [1, 2]})

        config = {
            "right_data": [{"id": 1}],
            "join_type": "inner",
            "left_on": ["id", "extra"],  # 2 keys
            "right_on": ["id"]  # 1 key
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(left_data, config)


class TestDuckDBWindowTransformer:
    """窗口函数转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = DuckDBWindowTransformer()

    def test_row_number(self, sample_orders_data: pl.DataFrame) -> None:
        """测试 ROW_NUMBER"""
        config = {
            "functions": [
                {
                    "function": "row_number",
                    "alias": "rn",
                    "order_by": ["order_id"]
                }
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "rn" in result.data.columns
        # row_number 应该从 1 开始
        rn_values = result.data["rn"].to_list()
        assert 1 in rn_values

    def test_rank_with_partition(self, sample_orders_data: pl.DataFrame) -> None:
        """测试带分区的 RANK"""
        config = {
            "functions": [
                {
                    "function": "rank",
                    "alias": "rank_in_customer",
                    "partition_by": ["customer_id"],
                    "order_by": ["quantity DESC"]
                }
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "rank_in_customer" in result.data.columns

    def test_running_sum(self, sample_orders_data: pl.DataFrame) -> None:
        """测试累计求和"""
        config = {
            "functions": [
                {
                    "function": "sum",
                    "column": "quantity",
                    "alias": "running_total",
                    "order_by": ["order_id"],
                    "frame": "ROWS UNBOUNDED PRECEDING"
                }
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "running_total" in result.data.columns

    def test_lag(self, sample_orders_data: pl.DataFrame) -> None:
        """测试 LAG"""
        config = {
            "functions": [
                {
                    "function": "lag",
                    "column": "quantity",
                    "alias": "prev_quantity",
                    "order_by": ["order_id"],
                    "offset": 1,
                    "default": 0
                }
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "prev_quantity" in result.data.columns
        # 第一行应该是 default 值 0
        assert result.data["prev_quantity"][0] == 0

    def test_lead(self, sample_orders_data: pl.DataFrame) -> None:
        """测试 LEAD"""
        config = {
            "functions": [
                {
                    "function": "lead",
                    "column": "quantity",
                    "alias": "next_quantity",
                    "order_by": ["order_id"],
                    "offset": 1
                }
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "next_quantity" in result.data.columns

    def test_ntile(self, sample_orders_data: pl.DataFrame) -> None:
        """测试 NTILE"""
        config = {
            "functions": [
                {
                    "function": "ntile",
                    "alias": "quartile",
                    "order_by": ["quantity DESC"],
                    "n": 4
                }
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "quartile" in result.data.columns
        # quartile 应该在 1-4 之间
        for q in result.data["quartile"].to_list():
            assert 1 <= q <= 4

    def test_multiple_window_functions(self, sample_orders_data: pl.DataFrame) -> None:
        """测试多个窗口函数"""
        config = {
            "functions": [
                {
                    "function": "row_number",
                    "alias": "rn",
                    "order_by": ["order_id"]
                },
                {
                    "function": "sum",
                    "column": "quantity",
                    "alias": "total_qty",
                    "partition_by": ["customer_id"],
                    "order_by": ["order_id"]
                },
                {
                    "function": "avg",
                    "column": "price",
                    "alias": "avg_price",
                    "partition_by": ["customer_id"],
                    "order_by": ["order_id"]
                }
            ]
        }

        result = self.transformer.execute(sample_orders_data, config)

        assert "rn" in result.data.columns
        assert "total_qty" in result.data.columns
        assert "avg_price" in result.data.columns

    def test_missing_functions_config(self, sample_orders_data: pl.DataFrame) -> None:
        """测试缺少 functions 配置"""
        with pytest.raises(TransformerConfigError):
            self.transformer.execute(sample_orders_data, {})

    def test_unsupported_function(self, sample_orders_data: pl.DataFrame) -> None:
        """测试不支持的函数"""
        config = {
            "functions": [
                {"function": "unsupported_func", "alias": "result"}
            ]
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(sample_orders_data, config)

    def test_missing_alias(self, sample_orders_data: pl.DataFrame) -> None:
        """测试缺少 alias"""
        config = {
            "functions": [
                {"function": "row_number", "order_by": ["order_id"]}
            ]
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(sample_orders_data, config)
