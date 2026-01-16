"""
QData Transformer 高级使用示例

展示更高级的转换器使用方式，包括：
- 自定义转换器
- 复杂的转换链
- 错误处理
- 配置序列化

由广东轻亿云软件科技有限公司开源
https://www.qeasy.cloud
"""

import json
import polars as pl
from qdata_transformer import (
    BaseTransformer,
    PolarsFieldMappingTransformer,
    TransformChain,
    TransformerRegistry,
    TransformerConfigError,
    TransformExecutionError,
)


# ============================================================
# 1. 自定义转换器示例
# ============================================================

@TransformerRegistry.register("data_validator")
class DataValidatorTransformer(BaseTransformer):
    """
    数据验证转换器

    验证数据并添加验证结果列。
    """

    name = "data_validator"
    description = "数据验证转换器"
    version = "1.0.0"

    def validate_config(self, config: dict) -> None:
        """验证配置"""
        if "rules" not in config:
            raise TransformerConfigError("配置缺少 'rules' 字段")

    def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        """执行验证转换"""
        result = data
        rules = config["rules"]

        for rule in rules:
            field = rule["field"]
            rule_type = rule["type"]
            target = rule.get("target", f"{field}_valid")

            if rule_type == "not_null":
                result = result.with_columns(
                    pl.col(field).is_not_null().alias(target)
                )
            elif rule_type == "positive":
                result = result.with_columns(
                    (pl.col(field) > 0).alias(target)
                )
            elif rule_type == "range":
                min_val = rule.get("min", float("-inf"))
                max_val = rule.get("max", float("inf"))
                result = result.with_columns(
                    ((pl.col(field) >= min_val) & (pl.col(field) <= max_val)).alias(target)
                )
            elif rule_type == "regex":
                pattern = rule["pattern"]
                result = result.with_columns(
                    pl.col(field).str.contains(pattern).alias(target)
                )

        return result


def demo_custom_transformer():
    """演示自定义转换器"""
    print("\n" + "=" * 60)
    print("1. 自定义转换器示例")
    print("=" * 60)

    data = pl.DataFrame({
        "user_id": ["U001", "U002", "U003", None],
        "email": ["alice@example.com", "invalid-email", "bob@test.org", "charlie@demo.com"],
        "age": [25, -5, 150, 30],
        "score": [85.5, 92.0, 78.5, 88.0],
    })

    print("\n原始数据:")
    print(data)

    # 使用自定义验证转换器
    validator = TransformerRegistry.get("data_validator")
    config = {
        "rules": [
            {"field": "user_id", "type": "not_null", "target": "user_id_valid"},
            {"field": "age", "type": "range", "min": 0, "max": 120, "target": "age_valid"},
            {"field": "email", "type": "regex", "pattern": r"^[\w.-]+@[\w.-]+\.\w+$", "target": "email_valid"},
        ]
    }

    result = validator.execute(data, config)
    print("\n验证结果:")
    print(result.data)


# ============================================================
# 2. 复杂转换链示例
# ============================================================

def demo_complex_chain():
    """演示复杂转换链"""
    print("\n" + "=" * 60)
    print("2. 复杂转换链示例")
    print("=" * 60)

    # 模拟电商订单数据
    orders = pl.DataFrame({
        "order_id": [f"O{i:03d}" for i in range(1, 11)],
        "customer_id": ["C001", "C002", "C001", "C003", "C002", "C001", "C003", "C002", "C001", "C003"],
        "product_category": ["电子", "服装", "电子", "食品", "电子", "服装", "电子", "食品", "服装", "电子"],
        "quantity": [2, 1, 3, 5, 1, 2, 1, 3, 1, 2],
        "unit_price": [999.0, 199.0, 599.0, 29.9, 1299.0, 299.0, 799.0, 39.9, 159.0, 499.0],
        "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04",
                       "2024-01-05", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-07"],
    })

    print("\n原始订单数据:")
    print(orders)

    # 构建转换链
    chain = (
        TransformChain()
        # 步骤1: 计算订单金额
        .add("polars_field_mapping", {
            "mappings": [
                {"source": "order_id", "target": "order_id"},
                {"source": "customer_id", "target": "customer_id"},
                {"source": "product_category", "target": "category"},
                {"source": "quantity", "target": "qty"},
                {"source": "unit_price", "target": "price"},
                {
                    "source": ["quantity", "unit_price"],
                    "target": "amount",
                    "transform": "expression",
                    "params": {"expr": "quantity * unit_price"}
                },
                {"source": "order_date", "target": "order_date"},
            ]
        }, "计算订单金额")
        # 步骤2: 按客户和品类聚合
        .add("duckdb_aggregation", {
            "group_by": ["customer_id", "category"],
            "aggregations": [
                {"field": "amount", "function": "sum", "alias": "total_amount"},
                {"field": "order_id", "function": "count", "alias": "order_count"},
                {"field": "amount", "function": "avg", "alias": "avg_order_value"},
            ],
            "order_by": ["total_amount DESC"],
        }, "客户品类聚合")
    )

    result = chain.execute(orders)
    print("\n转换链结果:")
    print(result.data)
    print(f"\n元数据: {json.dumps(result.metadata, indent=2, ensure_ascii=False)}")

    # 序列化转换链配置
    chain_config = chain.to_dict()
    print("\n转换链配置 (可保存为 JSON):")
    print(json.dumps(chain_config, indent=2, ensure_ascii=False))


# ============================================================
# 3. 错误处理示例
# ============================================================

def demo_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 60)
    print("3. 错误处理示例")
    print("=" * 60)

    data = pl.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30],
    })

    transformer = PolarsFieldMappingTransformer()

    # 示例1: 配置错误
    print("\n示例1: 处理配置错误")
    try:
        transformer.execute(data, {})  # 缺少 mappings
    except TransformerConfigError as e:
        print(f"捕获配置错误: {e}")
        print(f"错误详情: {e.to_dict()}")

    # 示例2: 列不存在
    print("\n示例2: 处理列不存在错误")
    try:
        transformer.execute(data, {
            "mappings": [
                {"source": "nonexistent_column", "target": "new_col"}
            ]
        })
    except TransformExecutionError as e:
        print(f"捕获执行错误: {e}")
        print(f"原始错误: {e.original_error}")

    # 示例3: 转换器不存在
    print("\n示例3: 处理转换器不存在错误")
    try:
        TransformerRegistry.get("unknown_transformer")
    except TransformExecutionError as e:
        print(f"捕获错误: {e}")


# ============================================================
# 4. 配置驱动示例
# ============================================================

def demo_config_driven():
    """演示配置驱动的转换"""
    print("\n" + "=" * 60)
    print("4. 配置驱动示例")
    print("=" * 60)

    # 从 JSON 配置创建转换链
    config_json = '''
    [
        {
            "transformer_name": "polars_field_mapping",
            "config": {
                "mappings": [
                    {"source": "name", "target": "user_name"},
                    {"source": "score", "target": "user_score"},
                    {"target": "processed", "transform": "constant", "params": {"value": true}}
                ]
            },
            "name": "字段映射",
            "enabled": true
        },
        {
            "transformer_name": "duckdb_aggregation",
            "config": {
                "group_by": [],
                "aggregations": [
                    {"field": "user_score", "function": "avg", "alias": "avg_score"},
                    {"field": "user_score", "function": "max", "alias": "max_score"},
                    {"field": "user_name", "function": "count", "alias": "total_users"}
                ]
            },
            "name": "全局聚合",
            "enabled": true
        }
    ]
    '''

    # 解析配置
    chain_config = json.loads(config_json)

    # 从配置创建转换链
    chain = TransformChain.from_dict(chain_config)

    # 准备数据
    data = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "score": [85, 92, 78, 88],
    })

    print("\n原始数据:")
    print(data)

    # 执行转换
    result = chain.execute(data)
    print("\n转换结果:")
    print(result.data)


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("=" * 60)
    print("QData Transformer 高级使用示例")
    print("广东轻亿云软件科技有限公司")
    print("https://www.qeasy.cloud")
    print("=" * 60)

    demo_custom_transformer()
    demo_complex_chain()
    demo_error_handling()
    demo_config_driven()

    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
