"""
QData Transformer 基本使用示例

展示最基本的转换器使用方式。
"""

import polars as pl
from qdata_transformer import (
    PolarsFieldMappingTransformer,
    DuckDBAggregationTransformer,
    TransformChain,
)


def main() -> None:
    """主函数"""
    
    # 1. 创建示例数据
    print("=" * 60)
    print("QData Transformer 基本使用示例")
    print("=" * 60)
    
    data = pl.DataFrame({
        "order_id": ["O001", "O002", "O003", "O004", "O005"],
        "customer_id": ["C001", "C002", "C001", "C003", "C002"],
        "quantity": [2, 3, 1, 5, 2],
        "price": [100.0, 200.0, 150.0, 100.0, 200.0],
    })
    
    print("\n原始数据:")
    print(data)
    
    # 2. 字段映射转换
    print("\n" + "-" * 40)
    print("步骤 1: 字段映射转换")
    print("-" * 40)
    
    mapping_transformer = PolarsFieldMappingTransformer()
    mapping_config = {
        "mappings": [
            {"source": "order_id", "target": "id"},
            {"source": "customer_id", "target": "customer"},
            {
                "source": ["quantity", "price"],
                "target": "amount",
                "transform": "expression",
                "params": {"expr": "quantity * price"},
            },
            {
                "target": "status",
                "transform": "constant",
                "params": {"value": "completed"},
            },
        ]
    }
    
    mapping_result = mapping_transformer.execute(data, mapping_config)
    print("\n映射结果:")
    print(mapping_result.data)
    print(f"输入行数: {mapping_result.input_rows}")
    print(f"输出行数: {mapping_result.output_rows}")
    
    # 3. 聚合转换
    print("\n" + "-" * 40)
    print("步骤 2: 聚合转换")
    print("-" * 40)
    
    agg_transformer = DuckDBAggregationTransformer()
    agg_config = {
        "group_by": ["customer"],
        "aggregations": [
            {"field": "amount", "function": "sum", "alias": "total_amount"},
            {"field": "amount", "function": "avg", "alias": "avg_amount"},
            {"field": "id", "function": "count", "alias": "order_count"},
        ],
        "order_by": ["total_amount DESC"],
    }
    
    agg_result = agg_transformer.execute(mapping_result.data, agg_config)
    print("\n聚合结果:")
    print(agg_result.data)
    
    # 4. 使用转换链
    print("\n" + "-" * 40)
    print("步骤 3: 使用转换链")
    print("-" * 40)
    
    chain = (
        TransformChain()
        .add("polars_field_mapping", mapping_config, "字段映射")
        .add("duckdb_aggregation", agg_config, "客户聚合")
    )
    
    chain_result = chain.execute(data)
    print("\n转换链结果:")
    print(chain_result.data)
    print(f"\n转换步骤数: {chain_result.metadata['chain_steps']}")
    print(f"步骤详情: {chain_result.metadata['step_results']}")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)
    print("\n更多信息请访问: https://www.qeasy.cloud")


if __name__ == "__main__":
    main()
