
# QData Transformer

<---

## 🌟 关于轻易云数据集成平台

<img src="https://qcdn.qeasy.cloud/static/qeasy-1.png" alt="数据集成平台" width="100%" />

> **数据集成，简单看得见**

轻易云是一款企业级数据集成平台，专注于帮助企业快速、高效地打通各类业务系统之间的数据通道。

### ✨ 平台亮点

- 🔌 **即插即用**：无需复杂开发，配置即连接，支持 500+ 主流应用系统对接
- 👁️ **全程可视**：数据流动、转换过程、执行状态一目了然，如同"物流跟踪"般清晰透明
- ⚡ **高性能引擎**：基于 Polars + DuckDB 构建，轻松处理百万级数据转换
- 🛡️ **企业级可靠**：完善的错误处理、重试机制与监控告警，保障数据安全稳定
- 🧩 **灵活扩展**：插件化架构设计，支持自定义转换器与数据源扩展

### 🎯 适用场景

| 场景 | 描述 |
|------|------|
| **系统集成** | ERP、CRM、WMS 等企业系统间的数据同步 |
| **数据迁移** | 跨平台、跨数据库的数据迁移与转换 |
| **ETL 流程** | 构建灵活的数据抽取、转换、加载流程 |
| **实时同步** | 业务数据的实时或准实时同步 |

**QData Transformer** 是轻易云数据集成平台的核心数据转换引擎，现已开源，助力开发者构建高效的数据处理流程。enter">
  <a href="https://www.qeasy.cloud">
    <img src="https://qcdn.qeasy.cloud/static/logo.svg" alt="轻易云" height="60">
  </a>
</p>

<p align="center">
  <strong>高性能、可扩展的数据转换引擎</strong>
</p>

<p align="center">
  由 <a href="https://www.qeasy.cloud">广东轻亿云软件科技有限公司</a> 开发<br>
  「轻易云数据集成平台」核心组件
</p>
<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.pola.rs/"><img src="https://img.shields.io/badge/Powered%20by-Polars-orange.svg" alt="Polars"></a>
  <a href="https://duckdb.org/"><img src="https://img.shields.io/badge/Powered%20by-DuckDB-green.svg" alt="DuckDB"></a>
  <a href="https://badge.fury.io/py/qdata-transformer"><img src="https://badge.fury.io/py/qdata-transformer.svg" alt="PyPI version"></a>
  <a href="https://github.com/vincent067/qdata-transformer"><img src="https://img.shields.io/github/stars/vincent067/qdata-transformer?style=social" alt="GitHub stars"></a>
</p>

---

## 🌟 关于轻易云数据集成平台

> **数据集成，简单看得见**  
> 从对接开始，到落地完成，全程清晰可控。  
> 无论是企业内部系统还是第三方平台，轻易云都能快速打通数据通道，无需复杂开发，配置即连接。数据怎么流动、如何转换、是否成功，全都清晰展示。就像“物流跟踪”一样，随时查看状态，让集成省时省力又省心。
<img src="https://qcdn.qeasy.cloud/static/qeasy-1.png" alt="数据集成平台" width="100%" />

**QData Transformer** 是轻易云数据集成平台的核心数据转换引擎，现已开源，助力开发者构建高效的数据处理流程。

---

## 📖 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [安装](#安装)
- [核心概念](#核心概念)
- [使用指南](./docs/usage.md)
- [示例](./examples/)
- [性能](./docs/performance.md)
- [API 参考](./docs/api.md)
- [贡献](./CONTRIBUTING.md)
- [许可证](#许可证)

---

## 特性

- 🚀 **高性能**：基于 Polars + DuckDB，提供卓越的数据处理性能  
- 🔧 **可扩展**：插件化架构，易于添加自定义转换器  
- 🔗 **可组合**：支持转换链，构建复杂的数据处理流程  
- 🛡️ **类型安全**：完整的类型注解，支持 mypy 静态检查  
- 📊 **丰富的转换器**：内置字段映射、数据聚合、SQL 查询等多种转换器  
- 🎯 **简单易用**：清晰的 API 设计，快速上手  
- 📈 **可观测**：内置监控和性能分析工具  

---

## 🚀 快速开始

### 安装

```bash
pip install qdata-transformer
```

### 第一个转换

```python
import polars as pl
from qdata_transformer import PolarsFieldMappingTransformer

data = pl.DataFrame({
    "order_id": ["O001", "O002"],
    "quantity": [2, 3],
    "price": [100.0, 200.0]
})

transformer = PolarsFieldMappingTransformer()
config = {
    "mappings": [
        {"source": "order_id", "target": "id"},
        {"source": ["quantity", "price"], "target": "amount",
         "transform": "expression", "params": {"expr": "quantity * price"}},
        {"target": "status", "transform": "constant", "params": {"value": "completed"}}
    ]
}

result = transformer.execute(data, config)
print(result.data)
```

---

## 📦 安装

```bash
# PyPI 安装
pip install qdata-transformer

# 源码安装
git clone https://github.com/vincent067/qdata-transformer.git
cd qdata-transformer
pip install -e .

# 开发依赖
pip install qdata-transformer[dev]
```

---

## 核心概念

| 概念 | 描述 |
|------|------|
| **转换器（Transformer）** | 数据处理的基本单元，实现特定转换逻辑 |
| **转换链（TransformChain）** | 多个转换器按顺序执行，构建复杂流程 |
| **注册中心（Registry）** | 管理所有可用转换器，支持动态发现与实例化 |

---

## 示例与文档

- 完整用法示例请见 [`./examples/`](./examples/)
- 详细文档请见 [`./docs/`](./docs/)

---

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证开源。

---

## 🏢 关于轻易云数据集成平台

**广东轻亿云软件科技有限公司**  
专注数据集成与处理，提供企业级 ETL/ELT 解决方案  
🌐 官网：[https://www.qeasy.cloud](https://www.qeasy.cloud)  
📧 联系：opensource@qeasy.cloud

---

*Powered by [广东轻亿云软件科技有限公司](https://www.qeasy.cloud)*
