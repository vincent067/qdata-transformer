# Changelog

本文件记录所有重要的变更�?

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)�?
本项目遵�?[语义化版本](https://semver.org/lang/zh-CN/)�?

## [Unreleased]

### Added
- 即将发布的新功能

## [1.0.0] - 2024-01-01

### Added
- 初始版本发布
- `PolarsFieldMappingTransformer` - Polars 1:1 字段映射转换�?
  - 支持直接映射、类型转换、常量值、表达式计算
  - 支持日期时间格式�?
- `PolarsMultiMappingTransformer` - Polars 1N:1N 批量映射转换�?
  - 支持嵌套字段访问
  - 支持数组展开 (explode)
  - 支持条件过滤
  - 支持 coalesce �?concat 操作
- `DuckDBAggregationTransformer` - DuckDB SQL 聚合转换�?
  - 支持 GROUP BY 聚合
  - 支持多种聚合函数：count, sum, avg, min, max, median �?
  - 支持 HAVING 过滤
  - 支持 ORDER BY 排序
- `DuckDBSQLTransformer` - DuckDB 自定�?SQL 转换�?
  - 支持任意 SQL 查询
- `TransformChain` - 转换�?
  - 支持串联多个转换�?
  - 支持序列化和反序列化
  - 支持步骤启用/禁用
- `TransformerRegistry` - 转换器注册中�?
  - 支持装饰器注�?
  - 支持编程式注�?
  - 单例模式管理转换器实�?
- 完整的异常体�?
- 完整的类型注解，支持 mypy
- 完整的单元测�?

### Security
- 无已知安全问�?

[Unreleased]: https://github.com/vincent067/qdata-transformer/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/vincent067/qdata-transformer/releases/tag/v1.0.0
