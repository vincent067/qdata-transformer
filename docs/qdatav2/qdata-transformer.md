# qdata-transformer

> **组件名**: qdata-transformer  
> **版本**: 0.1.0 (设计中)  
> **状态**: 设计规划初稿  
> **最后更新**: 2026-01-14

---

## 一、组件定位

`qdata-transformer` 是 QDataV2 的**数据转换引擎**，提供高性能、可扩展的数据转换能力。

### 1.1 核心职责

- **字段映射**：1:1、1:N、N:1 字段映射转换
- **数据聚合**：分组、求和、计数等聚合操作
- **数据拆分**：一条数据拆分为多条
- **表达式计算**：复杂的字段计算和表达式求值
- **数据过滤**：条件过滤、去重

### 1.2 设计目标

| 目标 | 说明 |
|------|------|
| **高性能** | 基于 Polars 的向量化计算 |
| **SQL友好** | 基于 DuckDB 的 SQL 聚合能力 |
| **可扩展** | 转换器插件机制，支持自定义 |
| **类型安全** | 完整类型注解，支持 mypy |

### 1.3 技术选型

| 技术 | 用途 | 选型理由 |
|------|------|---------|
| **Polars** | 数据转换核心 | 高性能、内存高效、API友好 |
| **DuckDB** | SQL聚合查询 | 嵌入式、SQL标准、高性能 |
| **Pydantic** | 配置验证 | 类型安全、验证完善 |

---

## 二、核心类设计

### 2.1 类图概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      qdata-transformer 核心类                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    BaseTransformer (ABC)                         │   │
│  │  ───────────────────────────────────────────────────────────── │   │
│  │  + transform(data, config) -> DataFrame                         │   │
│  │  + validate_config(config) -> bool                              │   │
│  │  + get_input_schema() -> Schema                                 │   │
│  │  + get_output_schema() -> Schema                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  ▲                                      │
│          ┌───────────────────────┼───────────────────────┐             │
│          │                       │                       │             │
│  ┌───────────────┐      ┌───────────────┐      ┌───────────────┐      │
│  │PolarsTransformer│    │DuckDBTransformer│    │CustomTransformer│     │
│  │               │      │               │      │               │      │
│  │• 字段映射     │      │• SQL聚合      │      │• 自定义逻辑   │      │
│  │• 数据过滤     │      │• 窗口函数     │      │• Python脚本   │      │
│  │• 列运算       │      │• JOIN操作     │      │               │      │
│  └───────────────┘      └───────────────┘      └───────────────┘      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   TransformerRegistry                            │   │
│  │  ───────────────────────────────────────────────────────────── │   │
│  │  + register(name, transformer_class)                            │   │
│  │  + get(name) -> BaseTransformer                                 │   │
│  │  + list_transformers() -> List[str]                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    TransformChain                                │   │
│  │  ───────────────────────────────────────────────────────────── │   │
│  │  + add(transformer, config)                                     │   │
│  │  + execute(data) -> DataFrame                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心类定义

#### 2.2.1 BaseTransformer（转换器基类）

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import polars as pl
from pydantic import BaseModel

class TransformConfig(BaseModel):
    """转换配置基类"""
    pass

class BaseTransformer(ABC):
    """转换器抽象基类"""
    
    name: str = "base"
    description: str = ""
    
    @abstractmethod
    def transform(
        self, 
        data: pl.DataFrame, 
        config: Dict[str, Any]
    ) -> pl.DataFrame:
        """
        执行数据转换
        
        Args:
            data: 输入数据（Polars DataFrame）
            config: 转换配置
            
        Returns:
            转换后的数据
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置有效性"""
        return True
    
    def get_input_schema(self) -> Optional[Dict]:
        """获取输入数据 Schema"""
        return None
    
    def get_output_schema(self) -> Optional[Dict]:
        """获取输出数据 Schema"""
        return None
    
    def pre_transform(self, data: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
        """转换前处理（可覆盖）"""
        return data
    
    def post_transform(self, data: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
        """转换后处理（可覆盖）"""
        return data
    
    def execute(self, data: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
        """执行完整转换流程"""
        self.validate_config(config)
        data = self.pre_transform(data, config)
        data = self.transform(data, config)
        data = self.post_transform(data, config)
        return data
```

#### 2.2.2 PolarsFieldMappingTransformer（字段映射转换器）

```python
class FieldMappingConfig(BaseModel):
    """字段映射配置"""
    mappings: List[Dict[str, Any]]
    
    class Config:
        extra = "forbid"

class PolarsFieldMappingTransformer(BaseTransformer):
    """基于 Polars 的字段映射转换器"""
    
    name = "polars_field_mapping"
    description = "高性能字段映射转换"
    
    def transform(self, data: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
        """执行字段映射"""
        cfg = FieldMappingConfig(**config)
        expressions = []
        
        for mapping in cfg.mappings:
            source = mapping.get("source")
            target = mapping["target"]
            transform_type = mapping.get("transform")
            params = mapping.get("params", {})
            
            if source is None:
                # 常量值
                if transform_type == "constant":
                    expressions.append(pl.lit(params["value"]).alias(target))
            elif isinstance(source, list):
                # 多字段运算
                if transform_type == "expression":
                    expr_str = params["expr"]
                    # 简单表达式解析（实际应使用表达式引擎）
                    expr = self._parse_expression(expr_str, source)
                    expressions.append(expr.alias(target))
            else:
                # 单字段映射
                if transform_type is None:
                    expressions.append(pl.col(source).alias(target))
                elif transform_type == "datetime_format":
                    expressions.append(
                        pl.col(source)
                        .str.strptime(pl.Datetime, params.get("input_format", "%Y-%m-%d"))
                        .alias(target)
                    )
                elif transform_type == "cast":
                    dtype = self._get_dtype(params.get("dtype", "str"))
                    expressions.append(pl.col(source).cast(dtype).alias(target))
        
        return data.select(expressions)
    
    def _parse_expression(self, expr_str: str, columns: List[str]) -> pl.Expr:
        """解析简单表达式"""
        # 简单实现，实际应使用表达式引擎
        if "*" in expr_str:
            parts = expr_str.split("*")
            return pl.col(parts[0].strip()) * pl.col(parts[1].strip())
        elif "+" in expr_str:
            parts = expr_str.split("+")
            return pl.col(parts[0].strip()) + pl.col(parts[1].strip())
        else:
            return pl.col(columns[0])
    
    def _get_dtype(self, dtype_str: str):
        """获取 Polars 数据类型"""
        dtype_map = {
            "str": pl.Utf8,
            "int": pl.Int64,
            "float": pl.Float64,
            "bool": pl.Boolean,
            "date": pl.Date,
            "datetime": pl.Datetime,
        }
        return dtype_map.get(dtype_str, pl.Utf8)
```

#### 2.2.3 DuckDBAggregationTransformer（聚合转换器）

```python
import duckdb

class AggregationConfig(BaseModel):
    """聚合配置"""
    group_by: List[str]
    aggregations: List[Dict[str, str]]
    having: Optional[str] = None

class DuckDBAggregationTransformer(BaseTransformer):
    """基于 DuckDB 的 SQL 聚合转换器"""
    
    name = "duckdb_aggregation"
    description = "SQL 聚合转换"
    
    def transform(self, data: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
        """执行 SQL 聚合"""
        cfg = AggregationConfig(**config)
        
        # 构建 SQL
        select_parts = list(cfg.group_by)
        for agg in cfg.aggregations:
            func = agg["function"]
            field = agg["field"]
            alias = agg.get("alias", f"{func}_{field}")
            select_parts.append(f"{func}({field}) AS {alias}")
        
        sql = f"""
            SELECT {', '.join(select_parts)}
            FROM data
            GROUP BY {', '.join(cfg.group_by)}
        """
        
        if cfg.having:
            sql += f" HAVING {cfg.having}"
        
        # 执行查询
        result = duckdb.query(sql).pl()
        return result

class DuckDBSQLTransformer(BaseTransformer):
    """通用 SQL 转换器"""
    
    name = "duckdb_sql"
    description = "自定义 SQL 转换"
    
    def transform(self, data: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
        """执行自定义 SQL"""
        sql = config.get("sql", "SELECT * FROM data")
        result = duckdb.query(sql).pl()
        return result
```

#### 2.2.4 TransformerRegistry（注册中心）

```python
from typing import Type

class TransformerRegistry:
    """转换器注册中心"""
    
    _instance = None
    _transformers: Dict[str, Type[BaseTransformer]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str = None):
        """注册转换器装饰器"""
        def decorator(transformer_class: Type[BaseTransformer]):
            key = name or transformer_class.name
            cls._transformers[key] = transformer_class
            return transformer_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> BaseTransformer:
        """获取转换器实例"""
        if name not in cls._transformers:
            raise KeyError(f"Transformer not found: {name}")
        return cls._transformers[name]()
    
    @classmethod
    def list_transformers(cls) -> List[str]:
        """列出所有注册的转换器"""
        return list(cls._transformers.keys())

# 注册内置转换器
@TransformerRegistry.register("polars_field_mapping")
class _PolarsFieldMappingTransformer(PolarsFieldMappingTransformer):
    pass

@TransformerRegistry.register("duckdb_aggregation")
class _DuckDBAggregationTransformer(DuckDBAggregationTransformer):
    pass

@TransformerRegistry.register("duckdb_sql")
class _DuckDBSQLTransformer(DuckDBSQLTransformer):
    pass
```

#### 2.2.5 TransformChain（转换链）

```python
@dataclass
class TransformStep:
    """转换步骤"""
    transformer: str
    config: Dict[str, Any]
    name: Optional[str] = None

class TransformChain:
    """转换链：串联多个转换器"""
    
    def __init__(self):
        self.steps: List[TransformStep] = []
        self.registry = TransformerRegistry()
    
    def add(
        self, 
        transformer: str, 
        config: Dict[str, Any],
        name: Optional[str] = None
    ) -> "TransformChain":
        """添加转换步骤"""
        self.steps.append(TransformStep(
            transformer=transformer,
            config=config,
            name=name
        ))
        return self
    
    def execute(self, data: pl.DataFrame) -> pl.DataFrame:
        """执行转换链"""
        result = data
        for step in self.steps:
            transformer = self.registry.get(step.transformer)
            result = transformer.execute(result, step.config)
        return result
    
    def to_dict(self) -> List[Dict]:
        """序列化为字典"""
        return [
            {
                "transformer": step.transformer,
                "config": step.config,
                "name": step.name
            }
            for step in self.steps
        ]
    
    @classmethod
    def from_dict(cls, steps: List[Dict]) -> "TransformChain":
        """从字典反序列化"""
        chain = cls()
        for step in steps:
            chain.add(
                transformer=step["transformer"],
                config=step["config"],
                name=step.get("name")
            )
        return chain
```

---

## 三、内置转换器

### 3.1 转换器清单

| 转换器 | 类型 | 功能 | 依赖 |
|--------|------|------|------|
| `polars_field_mapping` | Polars | 字段映射 | polars |
| `polars_filter` | Polars | 数据过滤 | polars |
| `polars_column_ops` | Polars | 列运算 | polars |
| `duckdb_aggregation` | DuckDB | 聚合运算 | duckdb |
| `duckdb_sql` | DuckDB | 自定义SQL | duckdb |
| `duckdb_join` | DuckDB | JOIN操作 | duckdb |
| `split` | 内置 | 数据拆分 | - |
| `merge` | 内置 | 数据合并 | - |
| `deduplicate` | 内置 | 数据去重 | polars |

### 3.2 配置示例

#### 字段映射配置

```yaml
transformer: polars_field_mapping
config:
  mappings:
    - source: order_id
      target: external_order_no
      
    - source: order_date
      target: created_at
      transform: datetime_format
      params:
        input_format: "%Y-%m-%d"
        
    - source: [qty, price]
      target: amount
      transform: expression
      params:
        expr: "qty * price"
        
    - target: status
      transform: constant
      params:
        value: PENDING
```

#### 聚合配置

```yaml
transformer: duckdb_aggregation
config:
  group_by: [customer_id, order_date]
  aggregations:
    - field: amount
      function: sum
      alias: total_amount
    - field: id
      function: count
      alias: order_count
  having: "total_amount > 1000"
```

#### 转换链配置

```yaml
chain:
  - transformer: polars_filter
    config:
      condition: "amount > 0"
    name: "过滤无效数据"
    
  - transformer: polars_field_mapping
    config:
      mappings: [...]
    name: "字段映射"
    
  - transformer: duckdb_aggregation
    config:
      group_by: [customer_id]
      aggregations: [...]
    name: "客户汇总"
```

---

## 四、API 设计

### 4.1 基础使用

```python
from qdata_transformer import TransformerRegistry, TransformChain
import polars as pl

# 获取转换器
transformer = TransformerRegistry.get("polars_field_mapping")

# 准备数据
data = pl.DataFrame({
    "order_id": ["O001", "O002"],
    "qty": [2, 3],
    "price": [100.0, 200.0]
})

# 执行转换
config = {
    "mappings": [
        {"source": "order_id", "target": "id"},
        {"source": ["qty", "price"], "target": "amount", "transform": "expression", "params": {"expr": "qty * price"}}
    ]
}
result = transformer.execute(data, config)
```

### 4.2 转换链使用

```python
from qdata_transformer import TransformChain

chain = (
    TransformChain()
    .add("polars_filter", {"condition": "qty > 0"})
    .add("polars_field_mapping", {"mappings": [...]})
    .add("duckdb_aggregation", {"group_by": ["customer_id"], "aggregations": [...]})
)

result = chain.execute(data)
```

### 4.3 自定义转换器

```python
from qdata_transformer import BaseTransformer, TransformerRegistry

@TransformerRegistry.register("custom_processor")
class CustomTransformer(BaseTransformer):
    name = "custom_processor"
    description = "自定义数据处理"
    
    def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        # 自定义逻辑
        return data.with_columns([
            pl.col("name").str.to_uppercase().alias("name_upper")
        ])
```

---

## 五、与主项目集成

### 5.1 在 DAG 节点中使用

```python
# 结合 qdata-dag-core 使用
from qdata_dag_core import Node, NodeType
from qdata_transformer import TransformChain

# 定义转换节点
transform_node = Node(
    id="transform_1",
    type=NodeType.ACTION,
    config={
        "transformer": "polars_field_mapping",
        "transform_config": {
            "mappings": [...]
        }
    }
)

# 执行节点
def execute_transform_node(node: Node, data: pl.DataFrame) -> pl.DataFrame:
    transformer_name = node.config["transformer"]
    transform_config = node.config["transform_config"]
    
    transformer = TransformerRegistry.get(transformer_name)
    return transformer.execute(data, transform_config)
```

### 5.2 在 Django 中使用

```python
# apps/integration/services.py
from qdata_transformer import TransformChain
import polars as pl

class DataTransformService:
    def transform_data(self, data: list, transform_config: list) -> list:
        """转换数据"""
        df = pl.DataFrame(data)
        chain = TransformChain.from_dict(transform_config)
        result = chain.execute(df)
        return result.to_dicts()
```

---

## 六、测试用例

```python
import pytest
import polars as pl
from qdata_transformer import (
    TransformerRegistry,
    TransformChain,
    PolarsFieldMappingTransformer
)

def test_field_mapping():
    transformer = PolarsFieldMappingTransformer()
    data = pl.DataFrame({
        "order_id": ["O001"],
        "qty": [2],
        "price": [100.0]
    })
    
    config = {
        "mappings": [
            {"source": "order_id", "target": "id"}
        ]
    }
    
    result = transformer.execute(data, config)
    assert "id" in result.columns
    assert result["id"][0] == "O001"

def test_transform_chain():
    chain = (
        TransformChain()
        .add("polars_field_mapping", {
            "mappings": [{"source": "a", "target": "b"}]
        })
    )
    
    data = pl.DataFrame({"a": [1, 2, 3]})
    result = chain.execute(data)
    assert "b" in result.columns

def test_registry():
    transformer = TransformerRegistry.get("polars_field_mapping")
    assert transformer is not None
    assert transformer.name == "polars_field_mapping"
```

---

## 七、项目结构

```
qdata_transformer/
├── __init__.py
├── base.py                 # BaseTransformer 基类
├── registry.py             # TransformerRegistry 注册中心
├── chain.py                # TransformChain 转换链
├── polars/                 # Polars 转换器
│   ├── __init__.py
│   ├── mapping.py          # 字段映射
│   ├── filter.py           # 数据过滤
│   └── column_ops.py       # 列运算
├── duckdb/                 # DuckDB 转换器
│   ├── __init__.py
│   ├── aggregation.py      # 聚合
│   ├── sql.py              # 自定义SQL
│   └── join.py             # JOIN操作
├── utils/                  # 工具函数
│   ├── __init__.py
│   └── schema.py           # Schema处理
├── py.typed
└── tests/
    ├── __init__.py
    ├── test_polars.py
    ├── test_duckdb.py
    └── test_chain.py
```

---

## 八、版本规划

| 版本 | 功能 | 状态 |
|------|------|------|
| 0.1.0 | 核心基类 + 字段映射 | 规划中 |
| 0.2.0 | Polars 转换器集合 | 规划中 |
| 0.3.0 | DuckDB 转换器集合 | 规划中 |
| 0.4.0 | 转换链 + 注册中心 | 规划中 |
| 1.0.0 | 稳定版本发布 | 规划中 |

---

## 九、相关文档

- [独立组件库总览](./README.md)
- [架构审计报告V2](../ARCHITECTURE-AUDIT-V2.md)
- [数据集成核心设计](../data-integration/README.md)

---

> **注意**：本文档为设计规划初稿，API 设计可能在实际开发过程中调整。
