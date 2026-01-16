"""
转换链测试
"""

import pytest
import polars as pl

from qdata_transformer import TransformChain, TransformerRegistry
from qdata_transformer.exceptions import TransformerNotFoundError


class TestTransformChain:
    """转换链测试类"""

    def test_simple_chain(self, sample_orders_data: pl.DataFrame) -> None:
        """测试简单转换链"""
        chain = (
            TransformChain()
            .add("polars_field_mapping", {
                "mappings": [
                    {"source": "order_id", "target": "id"},
                    {
                        "source": ["quantity", "price"],
                        "target": "amount",
                        "transform": "expression",
                        "params": {"expr": "quantity * price"},
                    },
                ]
            })
        )

        result = chain.execute(sample_orders_data)

        assert "id" in result.data.columns
        assert "amount" in result.data.columns

    def test_multi_step_chain(self, sample_orders_data: pl.DataFrame) -> None:
        """测试多步骤转换链"""
        chain = (
            TransformChain()
            .add("polars_field_mapping", {
                "mappings": [
                    {"source": "order_id", "target": "id"},
                    {"source": "customer_id", "target": "customer"},
                    {
                        "source": ["quantity", "price"],
                        "target": "amount",
                        "transform": "expression",
                        "params": {"expr": "quantity * price"},
                    },
                ]
            }, "字段映射")
            .add("duckdb_aggregation", {
                "group_by": ["customer"],
                "aggregations": [
                    {"field": "amount", "function": "sum", "alias": "total_amount"},
                    {"field": "id", "function": "count", "alias": "order_count"},
                ],
            }, "客户聚合")
        )

        result = chain.execute(sample_orders_data)

        assert "customer" in result.data.columns
        assert "total_amount" in result.data.columns
        assert "order_count" in result.data.columns
        assert result.metadata["chain_steps"] == 2

    def test_chain_serialization(self) -> None:
        """测试转换链序列化"""
        chain = (
            TransformChain()
            .add("polars_field_mapping", {"mappings": [{"source": "a", "target": "b"}]}, "step1")
            .add("duckdb_aggregation", {
                "aggregations": [{"field": "x", "function": "sum", "alias": "total"}]
            }, "step2")
        )

        # 序列化
        chain_dict = chain.to_dict()

        assert len(chain_dict) == 2
        assert chain_dict[0]["transformer_name"] == "polars_field_mapping"
        assert chain_dict[0]["name"] == "step1"
        assert chain_dict[1]["transformer_name"] == "duckdb_aggregation"

    def test_chain_deserialization(self, sample_orders_data: pl.DataFrame) -> None:
        """测试转换链反序列化"""
        steps = [
            {
                "transformer_name": "polars_field_mapping",
                "config": {
                    "mappings": [
                        {"source": "order_id", "target": "id"},
                    ]
                },
                "name": "映射",
                "enabled": True,
            }
        ]

        chain = TransformChain.from_dict(steps)
        result = chain.execute(sample_orders_data)

        assert "id" in result.data.columns

    def test_disabled_step(self, sample_orders_data: pl.DataFrame) -> None:
        """测试禁用的步骤"""
        chain = (
            TransformChain()
            .add("polars_field_mapping", {
                "mappings": [{"source": "order_id", "target": "id"}]
            }, enabled=True)
            .add("polars_field_mapping", {
                "mappings": [{"source": "id", "target": "new_id"}]
            }, enabled=False)  # 这个步骤被禁用
        )

        result = chain.execute(sample_orders_data)

        assert "id" in result.data.columns
        assert "new_id" not in result.data.columns

    def test_chain_length(self) -> None:
        """测试转换链长度"""
        chain = TransformChain()
        assert len(chain) == 0

        chain.add("polars_field_mapping", {"mappings": []})
        assert len(chain) == 1

        chain.add("duckdb_aggregation", {"aggregations": []})
        assert len(chain) == 2

    def test_invalid_transformer(self, sample_orders_data: pl.DataFrame) -> None:
        """测试无效的转换器名称"""
        chain = TransformChain().add("nonexistent_transformer", {})

        with pytest.raises(TransformerNotFoundError):
            chain.execute(sample_orders_data)
