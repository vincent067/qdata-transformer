# QData Transformer - 快速开始指南

> 由 [广东轻亿云软件科技有限公司](https://www.qeasy.cloud) 开源

## 🚀 5分钟快速上手

### 1. 安装

```bash
pip install qdata-transformer
```

或者从源码安装：

```bash
pip install polars duckdb pyarrow
pip install -e .
```

### 2. 运行第一个转换

创建文件 `first_transform.py`:

```python
import polars as pl
from qdata_transformer import (
    PolarsFieldMappingTransformer,
    TransformChain,
    TransformerRegistry,
)

# 1. 准备数据
data = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [50000, 60000, 70000]
})

print("原始数据:")
print(data)

# 2. 创建转换器
transformer = PolarsFieldMappingTransformer()

# 3. 配置转换
config = {
    "mappings": [
        # 重命名列
        {"source": "name", "target": "full_name"},
        # 计算新列
        {"source": ["age", "salary"], "target": "annual_income_millions",
         "transform": "expression", "params": {"expr": "salary / 1000000"}},
        # 添加常量列
        {"target": "processed_at", "transform": "constant",
         "params": {"value": "2024-01-15"}}
    ]
}

# 4. 执行转换
result = transformer.execute(data, config)

print("\n转换后数据:")
print(result.data)

print(f"\n处理统计:")
print(f"输入行数: {result.input_rows}")
print(f"输出行数: {result.output_rows}")
```

运行：
```bash
python first_transform.py
```

## 📚 核心概念速览

### 转换器 (Transformer)

转换器是数据处理的基本单元：

```python
# 使用内置转换器
transformer = PolarsFieldMappingTransformer()
result = transformer.execute(data, config)

# 从注册中心获取
transformer = TransformerRegistry.get("polars_field_mapping")
```

### 转换链 (TransformChain)

将多个转换器串联起来：

```python
chain = (
    TransformChain()
    .add("polars_field_mapping", {"mappings": [...]})
    .add("duckdb_aggregation", {"group_by": [...]})
)

result = chain.execute(data)
```

### 注册中心 (Registry)

管理所有转换器：

```python
# 注册转换器
TransformerRegistry.register_transformer("my_transformer", MyTransformer)

# 获取转换器
transformer = TransformerRegistry.get("my_transformer")

# 列出所有转换器
transformers = TransformerRegistry.list_transformers()
```

## 🎯 常用转换示例

### 1. 字段重命名和计算

```python
config = {
    "mappings": [
        {"source": "old_name", "target": "new_name"},
        {"source": ["qty", "price"], "target": "total",
         "transform": "expression", "params": {"expr": "qty * price"}}
    ]
}
```

### 2. 类型转换

```python
config = {
    "mappings": [
        {"source": "age_str", "target": "age_int",
         "transform": "cast", "params": {"dtype": "int"}},
        {"source": "date_str", "target": "date_col",
         "transform": "datetime", "params": {"format": "%Y-%m-%d"}}
    ]
}
```

### 3. 数据聚合

```python
from aggregation import DuckDBAggregationTransformer

config = {
    "group_by": ["category"],
    "aggregations": [
        {"field": "amount", "function": "sum", "alias": "total"},
        {"field": "amount", "function": "avg", "alias": "average"},
        {"field": "id", "function": "count", "alias": "count"}
    ]
}

result = DuckDBAggregationTransformer().execute(data, config)
```

### 4. 嵌套数据处理

```python
from multi_mapping import PolarsMultiMappingTransformer

config = {
    "mappings": [
        {"source": "customer.name", "target": "customer_name"},
        {"source": "items", "target": "item", "transform": "explode"}
    ]
}

result = PolarsMultiMappingTransformer().execute(data, config)
```

### 5. 自定义转换器

```python
@TransformerRegistry.register()
class MyTransformer(BaseTransformer):
    name = "my_transformer"
    
    def transform(self, data, config):
        # 实现转换逻辑
        return data.with_columns(pl.lit("processed").alias("status"))

# 使用
transformer = TransformerRegistry.get("my_transformer")
result = transformer.execute(data, config)
```

## 🔧 高级功能

### 数据过滤

```python
config = {
    "mappings": [...],
    "filter": {"condition": "amount > 1000"}
}
```

### 链式处理

```python
chain = TransformChain()
chain.add("mapping1", config1)
chain.add("mapping2", config2)
chain.add("aggregation", agg_config)

result = chain.execute(data)
print(f"处理步骤: {len(chain)}")
```

### 配置保存

```python
# 保存转换链
chain_dict = chain.to_dict()
with open("chain.json", "w") as f:
    json.dump(chain_dict, f)

# 加载转换链
with open("chain.json", "r") as f:
    chain_dict = json.load(f)
chain = TransformChain.from_dict(chain_dict)
```

## 📊 性能提示

### 最佳实践

1. **使用连接池** (对于 DuckDB 转换器)
   ```python
   # 连接池会自动管理
   ```

2. **批处理大数据**
   ```python
   for batch in data.iter_chunks(chunk_size=10000):
       result = transformer.execute(batch, config)
   ```

3. **缓存重复计算**
   ```python
   # 使用结果缓存避免重复计算
   ```

4. **选择合适的转换器**
   - 简单映射 → `PolarsFieldMappingTransformer`
   - 复杂聚合 → `DuckDBAggregationTransformer`
   - 嵌套数据 → `PolarsMultiMappingTransformer`

## 🐛 常见问题

### Q: 如何处理空值？
```python
# 在表达式中处理
config = {
    "mappings": [
        {"source": ["col1", "col2"], "target": "result",
         "transform": "expression", "params": {"expr": "col1.fill_null(0) + col2"}}
    ]
}
```

### Q: 如何调试转换链？
```python
# 查看每个步骤的结果
for i, step_result in enumerate(result.metadata['step_results']):
    print(f"步骤 {i+1}: {step_result['step_name']}")
    print(f"  输出行数: {step_result['output_rows']}")
```

### Q: 性能慢怎么办？
- 检查数据类型是否正确
- 使用批处理处理大数据
- 考虑使用并行处理
- 查看性能分析报告

### Q: 如何扩展功能？
```python
# 创建自定义转换器
class MyTransformer(BaseTransformer):
    name = "my_transformer"
    def transform(self, data, config):
        # 实现功能
        return data

# 注册并使用
TransformerRegistry.register_transformer("my_transformer", MyTransformer)
```

## 📖 学习路径

### 初学者
1. 阅读 README.md
2. 运行快速开始示例
3. 学习基础转换器使用
4. 尝试转换链

### 进阶用户
1. 阅读 API 文档
2. 学习自定义转换器
3. 了解性能优化技巧
4. 参与社区贡献

### 高级用户
1. 阅读架构设计文档
2. 实现插件系统
3. 贡献核心功能
4. 帮助维护项目

## 🆘 获取帮助

- 📖 **文档**: 查看项目中的文档文件
- 💬 **讨论**: 创建 GitHub Issue
- 🐛 **问题**: 报告 bug 或请求功能
- 📧 **邮件**: 联系维护者

## 🎉 下一步

1. **运行示例**: `python examples.py`
2. **阅读文档**: `README.md`, `FINAL_REPORT.md`
3. **尝试项目**: 应用到实际数据处理任务
4. **参与贡献**: 提交 Issue 和 Pull Request

---

**祝使用愉快！** 🎊
