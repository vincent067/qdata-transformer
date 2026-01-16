# 贡献指南

感谢您对 QData Transformer 的关注！我们欢迎所有形式的贡献�?

> **QData Transformer** �?[广东轻亿云软件科技有限公司](https://www.qeasy.cloud) 开源的"轻易云数据集成平�?核心组件�?

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设�?
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [Pull Request 流程](#pull-request-流程)
- [问题反馈](#问题反馈)

## 行为准则

参与本项目即表示您同意遵守我们的 [行为准则](CODE_OF_CONDUCT.md)。请在参与前阅读�?

## 如何贡献

### 报告 Bug

1. �?[Issues](https://github.com/vincent067/qdata-transformer/issues) 中搜索是否已存在相似问题
2. 如果没有，创建新 Issue，并提供�?
   - 清晰的问题描�?
   - 复现步骤
   - 期望行为 vs 实际行为
   - 环境信息（Python 版本、操作系统等�?
   - 如可能，提供最小复现代�?

### 功能建议

1. �?Issues 中搜索是否已有相似建�?
2. 创建�?Issue，说明：
   - 功能描述
   - 使用场景
   - 可能的实现方�?

### 提交代码

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 编写代码和测�?
4. 确保所有测试通过
5. 提交更改 (`git commit -m 'feat: add amazing feature'`)
6. 推送到分支 (`git push origin feature/amazing-feature`)
7. 创建 Pull Request

## 开发环境设�?

### 1. 克隆项目

```bash
git clone https://github.com/vincent067/qdata-transformer.git
cd qdata-transformer
```

### 2. 创建虚拟环境

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. 安装开发依�?

```bash
pip install -e ".[dev]"
```

### 4. 安装 pre-commit 钩子

```bash
pre-commit install
```

### 5. 运行测试

```bash
# 运行所有测�?
pytest

# 运行测试并生成覆盖率报告
pytest --cov=qdata_transformer --cov-report=html

# 运行特定测试文件
pytest tests/test_mapping.py

# 运行带标记的测试
pytest -m "not slow"
```

## 代码规范

### Python 代码风格

我们使用以下工具确保代码质量�?

- **Black**: 代码格式化（行宽 100�?
- **isort**: 导入排序
- **flake8**: 代码检�?
- **mypy**: 类型检�?

```bash
# 格式化代�?
black src tests

# 排序导入
isort src tests

# 代码检�?
flake8 src tests

# 类型检�?
mypy src
```

### 类型注解

所有公开 API 必须有完整的类型注解�?

```python
from typing import Any, Dict, List, Optional

def process_data(
    data: pl.DataFrame,
    config: Dict[str, Any],
    options: Optional[List[str]] = None,
) -> pl.DataFrame:
    """处理数据�?
    
    Args:
        data: 输入数据
        config: 配置字典
        options: 可选参数列�?
        
    Returns:
        处理后的数据
    """
    ...
```

### 文档字符�?

使用 Google 风格的文档字符串�?

```python
def transform(self, data: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
    """执行数据转换�?
    
    对输入数据执行配置指定的转换操作�?
    
    Args:
        data: 输入�?Polars DataFrame
        config: 转换配置字典，包�?mappings 等字�?
        
    Returns:
        转换后的 Polars DataFrame
        
    Raises:
        TransformerConfigError: 配置无效时抛�?
        TransformExecutionError: 转换执行失败时抛�?
        
    Example:
        >>> transformer = PolarsFieldMappingTransformer()
        >>> result = transformer.transform(df, {"mappings": [...]})
    """
    ...
```

## 提交规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 规范�?

### 提交类型

- `feat`: 新功�?
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行�?
- `refactor`: 重构（既不是新功能也不是 Bug 修复�?
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具变�?

### 提交格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 示例

```
feat(transformer): add support for nested field mapping

- Add nested field access via dot notation
- Support array explode operation
- Add filter condition support

Closes #123
```

## Pull Request 流程

### PR 检查清�?

- [ ] 代码遵循项目代码规范
- [ ] 所有测试通过
- [ ] 新功能有对应的测�?
- [ ] 文档已更新（如需要）
- [ ] CHANGELOG.md 已更新（如需要）
- [ ] 提交信息符合规范

### PR 描述模板

```markdown
## 描述
简要描述这�?PR 做了什�?

## 变更类型
- [ ] Bug 修复
- [ ] 新功�?
- [ ] 破坏性变�?
- [ ] 文档更新

## 测试
描述如何测试这些变更

## 相关 Issue
closes #xxx
```

### 代码审查

- 所�?PR 需要至少一位维护者审�?
- 审查者会关注代码质量、测试覆盖、文档完整�?
- 请及时响应审查意�?

## 问题反馈

- **Bug 报告**: [GitHub Issues](https://github.com/vincent067/qdata-transformer/issues)
- **功能建议**: [GitHub Discussions](https://github.com/vincent067/qdata-transformer/discussions)
- **安全问题**: 请发送邮件至 security@qeasy.cloud

## 致谢

感谢所有贡献者！您的每一份贡献都让这个项目变得更好�?

---

*[广东轻亿云软件科技有限公司](https://www.qeasy.cloud) - 让数据集成更简�?
