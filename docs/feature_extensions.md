# QData Transformer 功能扩展建议

## 1. 核心功能扩展

### 1.1 数据连接器 (高优先级) 🔵

**需求背景：**
当前库只支持内存中的 Polars DataFrame 作为输入，限制了数据来源的多样性。

**功能设计：**

```python
# 连接器基类
from abc import ABC, abstractmethod
from typing import Iterator, Any

class DataConnector(ABC):
    """数据连接器基类"""
    
    @abstractmethod
    def read(self, source: str, **kwargs) -> Iterator[pl.DataFrame]:
        """读取数据"""
        pass
    
    @abstractmethod
    def write(self, data: pl.DataFrame, destination: str, **kwargs) -> None:
        """写入数据"""
        pass

# 具体连接器实现
class CSVConnector(DataConnector):
    """CSV 文件连接器"""
    
    def read(self, source: str, **kwargs) -> Iterator[pl.DataFrame]:
        # 支持大文件分块读取
        chunk_size = kwargs.get('chunk_size', 10000)
        return pl.read_csv(source, batch_size=chunk_size)

class DatabaseConnector(DataConnector):
    """数据库连接器"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def read(self, source: str, **kwargs) -> Iterator[pl.DataFrame]:
        # 支持 SQL 查询
        query = kwargs.get('query', f'SELECT * FROM {source}')
        return pl.read_database(query, self.connection_string)

class S3Connector(DataConnector):
    """S3 连接器"""
    
    def __init__(self, bucket: str, access_key: str, secret_key: str):
        self.bucket = bucket
        # 初始化 S3 客户端
    
    def read(self, source: str, **kwargs) -> Iterator[pl.DataFrame]:
        # 从 S3 读取数据
        pass

# 连接器注册中心
class ConnectorRegistry:
    """连接器注册中心"""
    
    _connectors: dict[str, type[DataConnector]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(connector_class: type[DataConnector]):
            cls._connectors[name] = connector_class
            return connector_class
        return decorator
    
    @classmethod
    def get_connector(cls, name: str, **kwargs) -> DataConnector:
        if name not in cls._connectors:
            raise ValueError(f"连接器未找到: {name}")
        return cls._connectors[name](**kwargs)
```

### 1.2 表达式引擎增强 (高优先级) 🔵

**需求背景：**
当前表达式解析功能有限，仅支持简单的二元运算。

**功能设计：**

```python
# 增强的表达式引擎
from lark import Lark, Transformer, v_args

class ExpressionEngine:
    """表达式引擎"""
    
    # 扩展的语法支持
    grammar = r"""
        ?start: sum
        ?sum: product
            | sum "+" product   -> add
            | sum "-" product   -> sub
        ?product: atom
            | product "*" atom  -> mul
            | product "/" atom  -> div
            | product "%" atom  -> mod
        ?atom: NUMBER           -> number
             | CNAME           -> column
             | "(" sum ")"
             | atom "^" atom    -> power
             | "-" atom        -> neg
             | atom "!"        -> factorial
             | FUNCNAME "(" sum ")" -> func
        FUNCNAME: "sin" | "cos" | "tan" | "log" | "abs" | "round"
        %import common.NUMBER
        %import common.CNAME
        %import common.WS_INLINE
        %ignore WS_INLINE
    """
    
    def __init__(self):
        self.parser = Lark(self.grammar, parser='lalr', transformer=self.Transformer())
    
    @v_args(inline=True)
    class Transformer(Transformer):
        from operator import add, sub, mul, truediv as div, mod, pow, neg
        
        def number(self, token):
            return float(token)
        
        def column(self, token):
            return pl.col(str(token))
        
        def factorial(self, n):
            from math import factorial
            return factorial(int(n))
        
        def func(self, name, arg):
            func_map = {
                'sin': lambda x: x.sin(),
                'cos': lambda x: x.cos(),
                'tan': lambda x: x.tan(),
                'log': lambda x: x.log(),
                'abs': lambda x: x.abs(),
                'round': lambda x: x.round(),
            }
            return func_map.get(str(name), lambda x: x)(arg)
    
    def parse(self, expression: str) -> pl.Expr:
        """解析表达式为 Polars 表达式"""
        return self.parser.parse(expression)

# 集成到转换器
class EnhancedExpressionTransformer(BaseTransformer):
    """增强表达式转换器"""
    
    def __init__(self):
        self.expression_engine = ExpressionEngine()
    
    def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        expressions = []
        for mapping in config.get('mappings', []):
            if mapping.get('transform') == 'expression':
                expr_str = mapping['params']['expr']
                expr = self.expression_engine.parse(expr_str)
                expressions.append(expr.alias(mapping['target']))
        
        return data.with_columns(expressions)
```

### 1.3 缓存机制 (中优先级) 🟡

**需求背景：**
重复转换相同数据时性能可以进一步提升。

**功能设计：**

```python
import hashlib
import pickle
from functools import wraps
from typing import Any, Callable

class TransformCache:
    """转换缓存管理器"""
    
    def __init__(self, cache_size: int = 1000, ttl: int = 3600):
        self.cache_size = cache_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
    
    def _generate_key(self, data: pl.DataFrame, config: dict) -> str:
        """生成缓存键"""
        # 使用数据指纹和配置哈希
        data_hash = hashlib.md5(data.to_pandas().to_numpy().tobytes()).hexdigest()
        config_hash = hashlib.md5(pickle.dumps(config)).hexdigest()
        return f"{data_hash}_{config_hash}"
    
    def get(self, key: str) -> Any:
        """获取缓存"""
        if key in self.cache:
            # 检查 TTL
            import time
            if time.time() - self.access_times[key] < self.ttl:
                return self.cache[key]
            else:
                # 过期，移除缓存
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """设置缓存"""
        if len(self.cache) >= self.cache_size:
            # LRU 清理
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()

# 缓存装饰器
def with_cache(cache: TransformCache):
    """转换器缓存装饰器"""
    def decorator(transformer_class: type[BaseTransformer]):
        class CachedTransformer(transformer_class):
            def execute(self, data: pl.DataFrame, config: dict) -> TransformResult:
                cache_key = cache._generate_key(data, config)
                cached_result = cache.get(cache_key)
                
                if cached_result is not None:
                    return cached_result
                
                # 执行转换
                result = super().execute(data, config)
                
                # 缓存结果
                cache.put(cache_key, result)
                
                return result
        
        return CachedTransformer
    return decorator

# Redis 缓存后端
class RedisCacheBackend:
    """Redis 缓存后端"""
    
    def __init__(self, redis_url: str):
        import redis
        self.redis_client = redis.from_url(redis_url)
    
    def get(self, key: str) -> Any:
        value = self.redis_client.get(key)
        if value:
            return pickle.loads(value)
        return None
    
    def put(self, key: str, value: Any, ttl: int = 3600) -> None:
        self.redis_client.setex(key, ttl, pickle.dumps(value))
```

### 1.4 数据验证和模式检查 (中优先级) 🟡

**需求背景：**
需要确保输入数据符合预期的结构和类型。

**功能设计：**

```python
from pydantic import BaseModel, validator
from typing import Any, Dict, List, Union

class ColumnSchema(BaseModel):
    """列模式定义"""
    name: str
    dtype: str
    nullable: bool = True
    min_value: Union[int, float] = None
    max_value: Union[int, float] = None
    allowed_values: List[str] = None
    
    @validator('dtype')
    def validate_dtype(cls, v):
        allowed_dtypes = {'int64', 'float64', 'string', 'boolean', 'datetime'}
        if v not in allowed_dtypes:
            raise ValueError(f"不支持的数据类型: {v}")
        return v

class DataSchema(BaseModel):
    """数据模式定义"""
    columns: List[ColumnSchema]
    min_rows: int = 0
    max_rows: int = None
    
    def validate_dataframe(self, df: pl.DataFrame) -> bool:
        """验证 DataFrame"""
        # 检查列
        for col_schema in self.columns:
            if col_schema.name not in df.columns:
                raise ValueError(f"缺少必需的列: {col_schema.name}")
            
            # 检查数据类型
            expected_dtype = self._map_dtype(col_schema.dtype)
            actual_dtype = df[col_schema.name].dtype
            if str(actual_dtype) != expected_dtype:
                raise ValueError(f"列 {col_schema.name} 类型不匹配: 期望 {expected_dtype}, 实际 {actual_dtype}")
            
            # 检查可空性
            if not col_schema.nullable and df[col_schema.name].is_null().any():
                raise ValueError(f"列 {col_schema.name} 不允许空值")
            
            # 检查值范围
            if col_schema.min_value is not None:
                if (df[col_schema.name] < col_schema.min_value).any():
                    raise ValueError(f"列 {col_schema.name} 值小于最小值 {col_schema.min_value}")
            
            if col_schema.max_value is not None:
                if (df[col_schema.name] > col_schema.max_value).any():
                    raise ValueError(f"列 {col_schema.name} 值大于最大值 {col_schema.max_value}")
            
            # 检查允许值
            if col_schema.allowed_values is not None:
                invalid_values = set(df[col_schema.name].unique()) - set(col_schema.allowed_values)
                if invalid_values:
                    raise ValueError(f"列 {col_schema.name} 包含不允许的值: {invalid_values}")
        
        # 检查行数
        if len(df) < self.min_rows:
            raise ValueError(f"数据行数 {len(df)} 小于最小值 {self.min_rows}")
        
        if self.max_rows is not None and len(df) > self.max_rows:
            raise ValueError(f"数据行数 {len(df)} 大于最大值 {self.max_rows}")
        
        return True
    
    def _map_dtype(self, dtype: str) -> str:
        """映射数据类型"""
        dtype_map = {
            'int64': 'Int64',
            'float64': 'Float64',
            'string': 'Utf8',
            'boolean': 'Boolean',
            'datetime': 'Datetime'
        }
        return dtype_map.get(dtype, dtype)

# 集成到转换器
class SchemaValidatingTransformer(BaseTransformer):
    """模式验证转换器包装器"""
    
    def __init__(self, transformer: BaseTransformer, input_schema: DataSchema = None, output_schema: DataSchema = None):
        self.transformer = transformer
        self.input_schema = input_schema
        self.output_schema = output_schema
    
    def validate_input(self, data: pl.DataFrame) -> None:
        """验证输入"""
        if self.input_schema:
            self.input_schema.validate_dataframe(data)
    
    def validate_output(self, data: pl.DataFrame) -> None:
        """验证输出"""
        if self.output_schema:
            self.output_schema.validate_dataframe(data)
    
    def execute(self, data: pl.DataFrame, config: dict) -> TransformResult:
        """执行带验证的转换"""
        # 验证输入
        self.validate_input(data)
        
        # 执行转换
        result = self.transformer.execute(data, config)
        
        # 验证输出
        self.validate_output(result.data)
        
        return result
```

## 2. 高级功能扩展

### 2.1 机器学习数据预处理 (高优先级) 🔵

**需求背景：**
为机器学习工作流提供数据预处理能力。

**功能设计：**

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import numpy as np

class MLPreprocessorTransformer(BaseTransformer):
    """机器学习预处理转换器"""
    
    name = "ml_preprocessor"
    description = "机器学习数据预处理"
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.fitted = False
    
    def fit(self, data: pl.DataFrame, config: dict) -> None:
        """拟合预处理参数"""
        preprocessing_config = config.get('preprocessing', {})
        
        # 标准化
        for col in preprocessing_config.get('standardize', []):
            scaler = StandardScaler()
            scaler.fit(data[col].to_numpy().reshape(-1, 1))
            self.scalers[col] = scaler
        
        # 归一化
        for col in preprocessing_config.get('normalize', []):
            scaler = MinMaxScaler()
            scaler.fit(data[col].to_numpy().reshape(-1, 1))
            self.scalers[col] = scaler
        
        # 标签编码
        for col in preprocessing_config.get('label_encode', []):
            encoder = LabelEncoder()
            encoder.fit(data[col].to_numpy())
            self.encoders[col] = encoder
        
        self.fitted = True
    
    def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        """应用预处理"""
        if not self.fitted:
            raise ValueError("必须先调用 fit() 方法")
        
        result = data.clone()
        
        # 应用标准化/归一化
        for col, scaler in self.scalers.items():
            if col in result.columns:
                transformed = scaler.transform(result[col].to_numpy().reshape(-1, 1)).flatten()
                result = result.with_columns(pl.Series(transformed).alias(f"{col}_scaled"))
        
        # 应用标签编码
        for col, encoder in self.encoders.items():
            if col in result.columns:
                encoded = encoder.transform(result[col].to_numpy())
                result = result.with_columns(pl.Series(encoded).alias(f"{col}_encoded"))
        
        return result
    
    def execute(self, data: pl.DataFrame, config: dict) -> TransformResult:
        """执行转换（支持训练和推理模式）"""
        mode = config.get('mode', 'transform')
        
        if mode == 'fit':
            self.fit(data, config)
            return TransformResult(data=data, input_rows=len(data), output_rows=len(data))
        else:
            result = self.transform(data, config)
            return TransformResult(data=result, input_rows=len(data), output_rows=len(result))

# 特征工程转换器
class FeatureEngineeringTransformer(BaseTransformer):
    """特征工程转换器"""
    
    name = "feature_engineering"
    description = "特征工程转换"
    
    def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        features_config = config.get('features', [])
        result = data
        
        for feature_config in features_config:
            feature_type = feature_config['type']
            
            if feature_type == 'polynomial':
                result = self._create_polynomial_features(result, feature_config)
            elif feature_type == 'interaction':
                result = self._create_interaction_features(result, feature_config)
            elif feature_type == 'binning':
                result = self._create_binned_features(result, feature_config)
            elif feature_type == 'datetime':
                result = self._create_datetime_features(result, feature_config)
        
        return result
    
    def _create_polynomial_features(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        """创建多项式特征"""
        columns = config['columns']
        degree = config.get('degree', 2)
        
        result = data
        for col in columns:
            for d in range(2, degree + 1):
                result = result.with_columns(
                    (pl.col(col) ** d).alias(f"{col}_pow{d}")
                )
        
        return result
    
    def _create_interaction_features(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        """创建交互特征"""
        interactions = config['interactions']
        
        result = data
        for interaction in interactions:
            col1, col2 = interaction['columns']
            result = result.with_columns(
                (pl.col(col1) * pl.col(col2)).alias(f"{col1}_{col2}_interaction")
            )
        
        return result
    
    def _create_binned_features(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        """创建分箱特征"""
        column = config['column']
        bins = config['bins']
        labels = config.get('labels')
        
        result = data
        binned = pl.cut(pl.col(column), bins=bins, labels=labels)
        result = result.with_columns(binned.alias(f"{column}_binned"))
        
        return result
    
    def _create_datetime_features(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        """创建日期时间特征"""
        column = config['column']
        features = config.get('features', ['year', 'month', 'day', 'weekday'])
        
        result = data
        col = pl.col(column)
        
        feature_map = {
            'year': col.dt.year(),
            'month': col.dt.month(),
            'day': col.dt.day(),
            'weekday': col.dt.weekday(),
            'hour': col.dt.hour(),
            'minute': col.dt.minute(),
            'quarter': col.dt.quarter(),
            'is_weekend': col.dt.weekday() >= 5
        }
        
        for feature in features:
            if feature in feature_map:
                result = result.with_columns(
                    feature_map[feature].alias(f"{column}_{feature}")
                )
        
        return result
```

### 2.2 数据质量监控 (中优先级) 🟡

**需求背景：**
在数据转换过程中监控数据质量。

**功能设计：**

```python
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class DataQualityMetric:
    """数据质量指标"""
    metric_name: str
    column: str
    value: float
    threshold: float = None
    status: str = "PASS"  # PASS, WARNING, FAIL

class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self):
        self.metrics: List[DataQualityMetric] = []
    
    def check_completeness(self, data: pl.DataFrame, column: str) -> DataQualityMetric:
        """检查完整性（空值率）"""
        null_count = data[column].is_null().sum()
        total_count = len(data)
        null_rate = null_count / total_count
        
        return DataQualityMetric(
            metric_name="completeness",
            column=column,
            value=1 - null_rate,
            threshold=0.95,
            status="FAIL" if null_rate > 0.05 else "PASS"
        )
    
    def check_uniqueness(self, data: pl.DataFrame, column: str) -> DataQualityMetric:
        """检查唯一性"""
        unique_count = data[column].n_unique()
        total_count = len(data)
        uniqueness = unique_count / total_count
        
        return DataQualityMetric(
            metric_name="uniqueness",
            column=column,
            value=uniqueness,
            threshold=0.9,
            status="FAIL" if uniqueness < 0.9 else "PASS"
        )
    
    def check_validity(self, data: pl.DataFrame, column: str, validation_func) -> DataQualityMetric:
        """检查有效性"""
        valid_count = validation_func(data[column]).sum()
        total_count = len(data)
        validity = valid_count / total_count
        
        return DataQualityMetric(
            metric_name="validity",
            column=column,
            value=validity,
            threshold=0.99,
            status="FAIL" if validity < 0.99 else "PASS"
        )
    
    def check_consistency(self, data: pl.DataFrame, column1: str, column2: str) -> DataQualityMetric:
        """检查一致性"""
        # 示例：检查两列的相关性
        correlation = data[column1].corr(data[column2])
        
        return DataQualityMetric(
            metric_name="consistency",
            column=f"{column1}_{column2}",
            value=abs(correlation),
            threshold=0.7,
            status="WARNING" if abs(correlation) < 0.7 else "PASS"
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """生成质量报告"""
        total_metrics = len(self.metrics)
        passed_metrics = sum(1 for m in self.metrics if m.status == "PASS")
        failed_metrics = sum(1 for m in self.metrics if m.status == "FAIL")
        warning_metrics = sum(1 for m in self.metrics if m.status == "WARNING")
        
        return {
            "summary": {
                "total": total_metrics,
                "passed": passed_metrics,
                "failed": failed_metrics,
                "warning": warning_metrics,
                "score": passed_metrics / total_metrics if total_metrics > 0 else 1.0
            },
            "details": [
                {
                    "metric": m.metric_name,
                    "column": m.column,
                    "value": m.value,
                    "threshold": m.threshold,
                    "status": m.status
                }
                for m in self.metrics
            ]
        }

# 集成到转换器
class DataQualityTransformer(BaseTransformer):
    """数据质量检查转换器"""
    
    name = "data_quality_check"
    description = "数据质量检查"
    
    def __init__(self):
        self.monitor = DataQualityMonitor()
    
    def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        checks = config.get('checks', [])
        
        for check in checks:
            check_type = check['type']
            column = check['column']
            
            if check_type == 'completeness':
                metric = self.monitor.check_completeness(data, column)
                self.monitor.metrics.append(metric)
            elif check_type == 'uniqueness':
                metric = self.monitor.check_uniqueness(data, column)
                self.monitor.metrics.append(metric)
            elif check_type == 'validity':
                validation_func = check['validation_func']
                metric = self.monitor.check_validity(data, column, validation_func)
                self.monitor.metrics.append(metric)
            elif check_type == 'consistency':
                column2 = check['other_column']
                metric = self.monitor.check_consistency(data, column, column2)
                self.monitor.metrics.append(metric)
        
        # 数据质量报告可以作为元数据返回
        report = self.monitor.generate_report()
        print(f"数据质量报告: {report}")
        
        return data
    
    def execute(self, data: pl.DataFrame, config: dict) -> TransformResult:
        result = self.transform(data, config)
        
        # 生成质量报告
        report = self.monitor.generate_report()
        
        return TransformResult(
            data=result,
            input_rows=len(data),
            output_rows=len(result),
            metadata={
                "quality_report": report,
                "quality_score": report["summary"]["score"]
            }
        )
```

### 2.3 流式数据处理 (低优先级) 🟢

**需求背景：**
处理超大数据集或实时数据流。

**功能设计：**

```python
from typing import Iterator, Optional
import asyncio

class StreamTransformer:
    """流式数据转换器"""
    
    def __init__(self, transformers: List[BaseTransformer], buffer_size: int = 1000):
        self.transformers = transformers
        self.buffer_size = buffer_size
    
    def process_stream(self, stream: Iterator[pl.DataFrame], config: dict) -> Iterator[pl.DataFrame]:
        """处理数据流"""
        buffer = []
        
        for chunk in stream:
            buffer.append(chunk)
            
            if len(buffer) >= self.buffer_size:
                # 合并缓冲区数据
                combined = pl.concat(buffer)
                
                # 应用转换
                for transformer in self.transformers:
                    combined = transformer.execute(combined, config).data
                
                yield combined
                buffer = []
        
        # 处理剩余数据
        if buffer:
            combined = pl.concat(buffer)
            for transformer in self.transformers:
                combined = transformer.execute(combined, config).data
            yield combined

# 异步流处理
class AsyncStreamTransformer:
    """异步流式数据转换器"""
    
    def __init__(self, transformers: List[BaseTransformer], max_concurrent: int = 4):
        self.transformers = transformers
        self.max_concurrent = max_concurrent
        self.queue = asyncio.Queue(maxsize=100)
    
    async def process_stream_async(self, stream: Iterator[pl.DataFrame], config: dict) -> Iterator[pl.DataFrame]:
        """异步处理数据流"""
        # 创建任务队列
        tasks = []
        
        for chunk in stream:
            task = asyncio.create_task(self._process_chunk(chunk, config))
            tasks.append(task)
            
            if len(tasks) >= self.max_concurrent:
                # 等待任务完成
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    yield await task
                tasks = list(pending)
        
        # 处理剩余任务
        if tasks:
            done, _ = await asyncio.wait(tasks)
            for task in done:
                yield await task
    
    async def _process_chunk(self, chunk: pl.DataFrame, config: dict) -> pl.DataFrame:
        """处理数据块"""
        result = chunk
        for transformer in self.transformers:
            # 在单独的线程中执行转换
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: transformer.execute(result, config).data
            )
        return result
```

## 3. 工具和辅助功能

### 3.1 可视化工具 (中优先级) 🟡

**需求背景：**
帮助用户理解和调试数据转换过程。

**功能设计：**

```python
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph

class TransformVisualizer:
    """转换可视化工具"""
    
    def visualize_chain(self, chain: TransformChain) -> Digraph:
        """可视化转换链"""
        dot = Digraph(comment='Transform Chain')
        
        # 添加节点
        for i, step in enumerate(chain.steps):
            node_id = f"step_{i}"
            label = f"{step.name or step.transformer_name}\n{step.transformer_name}"
            dot.node(node_id, label, shape='box')
            
            # 添加边
            if i > 0:
                dot.edge(f"step_{i-1}", node_id)
        
        return dot
    
    def visualize_data_profile(self, data: pl.DataFrame, output_path: str) -> None:
        """可视化数据概况"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 数值列分布
        numeric_cols = [col for col in data.columns if data[col].dtype in (pl.Int64, pl.Float64)]
        if numeric_cols:
            data[numeric_cols].describe().to_pandas().plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Numeric Columns Statistics')
        
        # 缺失值热图
        missing_data = data.select([pl.col(col).is_null().sum() for col in data.columns])
        if missing_data.shape[1] > 0:
            sns.heatmap(missing_data.to_pandas(), annot=True, ax=axes[0, 1])
            axes[0, 1].set_title('Missing Values')
        
        # 相关性矩阵
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().to_pandas()
            sns.heatmap(corr_matrix, annot=True, ax=axes[1, 0])
            axes[1, 0].set_title('Correlation Matrix')
        
        # 数据类型分布
        dtype_counts = data.dtypes.value_counts()
        axes[1, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Data Types Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def visualize_transformation_impact(self, before: pl.DataFrame, after: pl.DataFrame, output_path: str) -> None:
        """可视化转换影响"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 行数变化
        axes[0].bar(['Before', 'After'], [len(before), len(after)])
        axes[0].set_title('Row Count')
        axes[0].set_ylabel('Number of Rows')
        
        # 列数变化
        axes[1].bar(['Before', 'After'], [len(before.columns), len(after.columns)])
        axes[1].set_title('Column Count')
        axes[1].set_ylabel('Number of Columns')
        
        # 内存使用变化
        before_memory = before.estimated_size()
        after_memory = after.estimated_size()
        axes[2].bar(['Before', 'After'], [before_memory, after_memory])
        axes[2].set_title('Memory Usage')
        axes[2].set_ylabel('Bytes')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
```

### 3.2 配置生成器 (低优先级) 🟢

**需求背景：**
帮助用户生成转换配置。

**功能设计：**

```python
class ConfigGenerator:
    """配置生成器"""
    
    @staticmethod
    def generate_field_mapping_config(data: pl.DataFrame, mapping_type: str = 'direct') -> dict:
        """生成字段映射配置"""
        mappings = []
        
        for col in data.columns:
            if mapping_type == 'direct':
                mappings.append({
                    "source": col,
                    "target": col
                })
            elif mapping_type == 'uppercase':
                mappings.append({
                    "source": col,
                    "target": col.upper()
                })
            elif mapping_type == 'snake_case':
                import re
                snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower()
                mappings.append({
                    "source": col,
                    "target": snake_case
                })
        
        return {"mappings": mappings}
    
    @staticmethod
    def generate_aggregation_config(group_by: List[str], agg_columns: List[str]) -> dict:
        """生成聚合配置"""
        aggregations = []
        
        for col in agg_columns:
            aggregations.extend([
                {"field": col, "function": "sum", "alias": f"{col}_sum"},
                {"field": col, "function": "avg", "alias": f"{col}_avg"},
                {"field": col, "function": "min", "alias": f"{col}_min"},
                {"field": col, "function": "max", "alias": f"{col}_max"}
            ])
        
        return {
            "group_by": group_by,
            "aggregations": aggregations
        }
    
    @staticmethod
    def generate_ml_preprocessing_config(data: pl.DataFrame, target_column: str = None) -> dict:
        """生成机器学习预处理配置"""
        config = {
            "preprocessing": {
                "standardize": [],
                "normalize": [],
                "label_encode": [],
                "one_hot_encode": []
            }
        }
        
        for col in data.columns:
            if col == target_column:
                continue
                
            dtype = data[col].dtype
            
            if dtype in (pl.Int64, pl.Float64):
                config["preprocessing"]["standardize"].append(col)
            elif dtype == pl.Utf8:
                unique_ratio = data[col].n_unique() / len(data)
                if unique_ratio < 0.1:  # 低基数分类变量
                    config["preprocessing"]["one_hot_encode"].append(col)
                else:  # 高基数分类变量
                    config["preprocessing"]["label_encode"].append(col)
        
        return config
```

## 4. 性能优化扩展

### 4.1 并行处理框架 (高优先级) 🔵

**需求背景：**
充分利用多核 CPU 提升处理性能。

**功能设计：**

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager
import multiprocessing as mp

class ParallelTransformer(BaseTransformer):
    """并行转换器"""
    
    name = "parallel_processor"
    description = "并行数据处理"
    
    def __init__(self, n_workers: int = None, strategy: str = 'process'):
        self.n_workers = n_workers or mp.cpu_count()
        self.strategy = strategy  # 'process' 或 'thread'
        self.executor_class = ProcessPoolExecutor if strategy == 'process' else ThreadPoolExecutor
    
    def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        """并行转换数据"""
        # 数据分区
        partitions = self._partition_data(data, self.n_workers)
        
        # 获取子转换器配置
        sub_transformer_config = config.get('sub_transformer', {})
        transformer_name = sub_transformer_config.get('name')
        transformer_config = sub_transformer_config.get('config', {})
        
        if not transformer_name:
            raise ValueError("必须指定子转换器名称")
        
        # 并行处理
        with self.executor_class(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(
                    self._process_partition,
                    partition,
                    transformer_name,
                    transformer_config
                )
                for partition in partitions
            ]
            
            results = [future.result() for future in futures]
        
        # 合并结果
        return pl.concat(results)
    
    def _partition_data(self, data: pl.DataFrame, n_partitions: int) -> List[pl.DataFrame]:
        """数据分区"""
        chunk_size = len(data) // n_partitions
        partitions = []
        
        for i in range(n_partitions):
            start = i * chunk_size
            end = start + chunk_size if i < n_partitions - 1 else len(data)
            partitions.append(data[start:end])
        
        return partitions
    
    def _process_partition(self, partition: pl.DataFrame, transformer_name: str, config: dict) -> pl.DataFrame:
        """处理数据分区"""
        # 在每个进程中重新获取转换器实例
        transformer = TransformerRegistry.get(transformer_name)
        result = transformer.execute(partition, config)
        return result.data

# 分布式处理
class DistributedTransformer(BaseTransformer):
    """分布式转换器"""
    
    name = "distributed_processor"
    description = "分布式数据处理"
    
    def __init__(self, cluster_address: str):
        self.cluster_address = cluster_address
        # 初始化分布式计算客户端（如 Dask、Ray）
    
    def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        """分布式转换"""
        # 将数据分发到集群
        # 在集群上执行转换
        # 收集结果
        pass
```

### 4.2 缓存系统 (中优先级) 🟡

**需求背景：**
避免重复计算，提升性能。

**功能设计：**

已在核心功能扩展中详细描述，此处补充缓存策略：

```python
class CacheStrategy:
    """缓存策略基类"""
    
    @abstractmethod
    def should_cache(self, data_size: int, computation_cost: float) -> bool:
        """判断是否应缓存"""
        pass

class SizeBasedCacheStrategy(CacheStrategy):
    """基于数据大小的缓存策略"""
    
    def __init__(self, max_data_size: int = 1000000):
        self.max_data_size = max_data_size
    
    def should_cache(self, data_size: int, computation_cost: float) -> bool:
        return data_size <= self.max_data_size

class CostBasedCacheStrategy(CacheStrategy):
    """基于计算成本的缓存策略"""
    
    def __init__(self, min_computation_cost: float = 1.0):
        self.min_computation_cost = min_computation_cost
    
    def should_cache(self, data_size: int, computation_cost: float) -> bool:
        return computation_cost >= self.min_computation_cost

class AdaptiveCacheStrategy(CacheStrategy):
    """自适应缓存策略"""
    
    def should_cache(self, data_size: int, computation_cost: float) -> bool:
        # 缓存效益 = 计算成本 / 数据大小
        benefit = computation_cost / (data_size + 1)
        return benefit > 0.001  # 阈值可配置
```

## 5. 监控和可观测性

### 5.1 性能监控 (高优先级) 🔵

**需求背景：**
监控转换性能，发现性能瓶颈。

**功能设计：**

```python
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Any
import json

@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float
    cpu_usage: float
    memory_peak: float
    memory_avg: float
    io_read_bytes: int
    io_write_bytes: int
    throughput: float  # rows/second
    
class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.process = psutil.Process()
    
    def profile(self, func: Callable) -> Callable:
        """性能分析装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 开始监控
            start_time = time.time()
            start_memory = self.process.memory_info().rss
            start_io = self.process.io_counters()
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 结束监控
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            end_io = self.process.io_counters()
            
            # 计算指标
            execution_time = end_time - start_time
            data_size = len(args[0]) if args else 0  # 假设第一个参数是数据
            throughput = data_size / execution_time if execution_time > 0 else 0
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                cpu_usage=self.process.cpu_percent(),
                memory_peak=max(start_memory, end_memory),
                memory_avg=(start_memory + end_memory) / 2,
                io_read_bytes=end_io.read_bytes - start_io.read_bytes,
                io_write_bytes=end_io.write_bytes - start_io.write_bytes,
                throughput=throughput
            )
            
            self.metrics_history.append(metrics)
            
            return result
        
        return wrapper
    
    def get_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.metrics_history:
            return {}
        
        avg_execution_time = sum(m.execution_time for m in self.metrics_history) / len(self.metrics_history)
        avg_throughput = sum(m.throughput for m in self.metrics_history) / len(self.metrics_history)
        peak_memory = max(m.memory_peak for m in self.metrics_history)
        
        return {
            "summary": {
                "total_operations": len(self.metrics_history),
                "avg_execution_time": avg_execution_time,
                "avg_throughput": avg_throughput,
                "peak_memory_mb": peak_memory / 1024 / 1024
            },
            "details": [
                {
                    "execution_time": m.execution_time,
                    "throughput": m.throughput,
                    "memory_peak_mb": m.memory_peak / 1024 / 1024,
                    "cpu_usage": m.cpu_usage
                }
                for m in self.metrics_history
            ]
        }

# 集成 Prometheus
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class PrometheusMonitor:
    """Prometheus 监控集成"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        
        # 指标定义
        self.execution_counter = Counter(
            'transformer_executions_total',
            'Total number of transformer executions',
            ['transformer_name', 'status']
        )
        
        self.execution_duration = Histogram(
            'transformer_execution_duration_seconds',
            'Execution duration in seconds',
            ['transformer_name']
        )
        
        self.memory_usage = Gauge(
            'transformer_memory_usage_bytes',
            'Memory usage in bytes',
            ['transformer_name']
        )
        
        self.throughput_gauge = Gauge(
            'transformer_throughput_rows_per_second',
            'Processing throughput in rows per second',
            ['transformer_name']
        )
        
        # 启动 HTTP 服务器
        start_http_server(port)
    
    def record_execution(self, transformer_name: str, duration: float, status: str, data_size: int):
        """记录执行指标"""
        self.execution_counter.labels(transformer_name=transformer_name, status=status).inc()
        self.execution_duration.labels(transformer_name=transformer_name).observe(duration)
        
        throughput = data_size / duration if duration > 0 else 0
        self.throughput_gauge.labels(transformer_name=transformer_name).set(throughput)
    
    def record_memory(self, transformer_name: str, memory_bytes: int):
        """记录内存使用"""
        self.memory_usage.labels(transformer_name=transformer_name).set(memory_bytes)
```

### 5.2 日志系统 (中优先级) 🟡

**需求背景：**
详细的日志记录便于调试和问题排查。

**功能设计：**

```python
import logging
import json
from typing import Any, Dict
from datetime import datetime

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str, level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 配置处理器
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_transform_start(self, transformer_name: str, data_size: int, config: Dict[str, Any]):
        """记录转换开始"""
        self.logger.info(
            "Transform started",
            extra={
                "event": "transform_start",
                "transformer": transformer_name,
                "data_size": data_size,
                "config": config
            }
        )
    
    def log_transform_end(self, transformer_name: str, duration: float, status: str, error: str = None):
        """记录转换结束"""
        extra = {
            "event": "transform_end",
            "transformer": transformer_name,
            "duration": duration,
            "status": status
        }
        
        if error:
            extra["error"] = error
            self.logger.error("Transform failed", extra=extra)
        else:
            self.logger.info("Transform completed", extra=extra)
    
    def log_data_quality_issue(self, issue_type: str, column: str, details: Dict[str, Any]):
        """记录数据质量问题"""
        self.logger.warning(
            "Data quality issue detected",
            extra={
                "event": "data_quality_issue",
                "type": issue_type,
                "column": column,
                "details": details
            }
        )
    
    def log_performance_warning(self, transformer_name: str, metric: str, value: float, threshold: float):
        """记录性能警告"""
        self.logger.warning(
            "Performance issue detected",
            extra={
                "event": "performance_warning",
                "transformer": transformer_name,
                "metric": metric,
                "value": value,
                "threshold": threshold
            }
        )

# JSON 格式的日志处理器
class JSONLogHandler(logging.Handler):
    """JSON 格式日志处理器"""
    
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
    
    def emit(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        
        # 添加额外信息
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        # 写入文件
        with open(self.filename, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

## 6. 总结

### 6.1 功能扩展优先级

| 功能 | 优先级 | 预计工作量 | 价值 |
|------|--------|-----------|------|
| 数据连接器 | 🔵 高 | 2-3 周 | ⭐⭐⭐⭐⭐ |
| 表达式引擎增强 | 🔵 高 | 1-2 周 | ⭐⭐⭐⭐⭐ |
| 机器学习预处理 | 🔵 高 | 2-3 周 | ⭐⭐⭐⭐ |
| 缓存机制 | 🟡 中 | 1-2 周 | ⭐⭐⭐⭐ |
| 数据质量监控 | 🟡 中 | 2-3 周 | ⭐⭐⭐⭐ |
| 并行处理框架 | 🟡 中 | 3-4 周 | ⭐⭐⭐⭐ |
| 可视化工具 | 🟡 中 | 2-3 周 | ⭐⭐⭐ |
| 性能监控 | 🔵 高 | 1-2 周 | ⭐⭐⭐⭐ |
| 流式处理 | 🟢 低 | 4-5 周 | ⭐⭐⭐ |

### 6.2 实施建议

**第一阶段 (1-2 个月)：**
1. 实现数据连接器（CSV、数据库）
2. 增强表达式引擎
3. 添加基础性能监控
4. 实现缓存机制

**第二阶段 (2-3 个月)：**
1. 添加机器学习预处理功能
2. 实现数据质量监控
3. 添加并行处理框架
4. 完善可视化工具

**第三阶段 (3-6 个月)：**
1. 添加更多连接器（S3、API 等）
2. 实现流式数据处理
3. 添加高级监控和告警
4. 持续优化性能

### 6.3 预期效果

通过功能扩展，QData Transformer 将从单一的数据转换库演变为：

1. **完整的数据处理平台** - 支持多种数据源和目标
2. **高性能计算引擎** - 支持并行和分布式处理
3. **智能数据管道** - 集成机器学习和数据质量监控
4. **可观测系统** - 完善的监控、日志和可视化能力

这将大大提升库的竞争力和实用性，使其成为数据工程领域的重要工具。
