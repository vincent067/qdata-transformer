"""
转换器基类

定义数据转换器的抽象基类和相关协议。
设计原则：
1. 类型安全：完整类型注解，支持 mypy
2. 可扩展：支持自定义转换器
3. 可组合：支持转换链
4. 高性能：基于 Polars 的向量化计算
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import polars as pl

from qdata_transformer.exceptions import TransformerConfigError, TransformExecutionError

if TYPE_CHECKING:
    from qdata_transformer.registry import TransformerRegistry


@dataclass
class TransformResult:
    """
    转换结果

    包含转换后的数据和相关元信息。
    """

    data: pl.DataFrame  # 转换后的数据
    input_rows: int = 0  # 输入行数
    output_rows: int = 0  # 输出行数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    @property
    def rows_changed(self) -> int:
        """行数变化"""
        return self.output_rows - self.input_rows


class BaseTransformer(ABC):
    """
    转换器抽象基类

    所有数据转换器必须继承此类并实现 transform 方法。

    使用示例：
        class MyTransformer(BaseTransformer):
            name = "my_transformer"
            description = "自定义转换器"

            def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
                return data.with_columns(...)

        transformer = MyTransformer()
        result = transformer.execute(df, {"key": "value"})

    属性：
        name: 转换器唯一标识
        description: 转换器描述
        version: 版本号
    """

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    version: ClassVar[str] = "1.0.0"

    @abstractmethod
    def transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """
        执行数据转换

        Args:
            data: 输入数据（Polars DataFrame）
            config: 转换配置

        Returns:
            转换后的数据

        Raises:
            TransformExecutionError: 转换执行失败时抛出
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置有效性

        子类可覆盖此方法实现配置验证逻辑。

        Args:
            config: 转换配置

        Raises:
            TransformerConfigError: 配置无效时抛出
        """
        pass

    def pre_transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """
        转换前处理

        子类可覆盖此方法实现预处理逻辑。

        Args:
            data: 输入数据
            config: 转换配置

        Returns:
            预处理后的数据
        """
        return data

    def post_transform(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> pl.DataFrame:
        """
        转换后处理

        子类可覆盖此方法实现后处理逻辑。

        Args:
            data: 转换后数据
            config: 转换配置

        Returns:
            后处理后的数据
        """
        return data

    def execute(
        self,
        data: pl.DataFrame,
        config: Dict[str, Any],
    ) -> TransformResult:
        """
        执行完整转换流程

        流程：验证配置 -> 预处理 -> 转换 -> 后处理

        Args:
            data: 输入数据
            config: 转换配置

        Returns:
            TransformResult 转换结果

        Raises:
            TransformerConfigError: 配置无效时抛出
            TransformExecutionError: 转换执行失败时抛出
        """
        input_rows = len(data)

        try:
            # 验证配置
            self.validate_config(config)

            # 预处理
            data = self.pre_transform(data, config)

            # 执行转换
            result_data = self.transform(data, config)

            # 后处理
            result_data = self.post_transform(result_data, config)

            return TransformResult(
                data=result_data,
                input_rows=input_rows,
                output_rows=len(result_data),
                metadata={
                    "transformer": self.name,
                    "version": self.version,
                },
            )

        except TransformerConfigError:
            raise
        except Exception as e:
            raise TransformExecutionError(
                message=str(e),
                transformer_name=self.name,
                original_error=e,
            ) from e

    def get_info(self) -> Dict[str, Any]:
        """
        获取转换器信息

        Returns:
            包含转换器元信息的字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }


@dataclass
class TransformStep:
    """
    转换步骤

    描述转换链中的一个步骤。
    """

    transformer_name: str  # 转换器名称
    config: Dict[str, Any]  # 转换配置
    name: str = ""  # 步骤名称（可选）
    enabled: bool = True  # 是否启用


class TransformChain:
    """
    转换链

    串联多个转换器，按顺序执行。

    使用示例：
        from qdata_transformer import TransformChain, TransformerRegistry

        chain = TransformChain(registry)
        chain.add("polars_field_mapping", {"mappings": [...]})
        chain.add("polars_filter", {"condition": "amount > 0"})

        result = chain.execute(data)
    """

    def __init__(self, registry: Optional["TransformerRegistry"] = None):
        """
        初始化转换链

        Args:
            registry: 转换器注册中心（可选，默认使用全局注册中心）
        """
        from qdata_transformer.registry import TransformerRegistry as DefaultRegistry

        self._registry = registry or DefaultRegistry()
        self._steps: List[TransformStep] = []

    def add(
        self,
        transformer_name: str,
        config: Dict[str, Any],
        name: str = "",
        enabled: bool = True,
    ) -> "TransformChain":
        """
        添加转换步骤

        Args:
            transformer_name: 转换器名称
            config: 转换配置
            name: 步骤名称
            enabled: 是否启用

        Returns:
            self，支持链式调用
        """
        self._steps.append(
            TransformStep(
                transformer_name=transformer_name,
                config=config,
                name=name,
                enabled=enabled,
            )
        )
        return self

    def execute(self, data: pl.DataFrame) -> TransformResult:
        """
        执行转换链

        Args:
            data: 输入数据

        Returns:
            TransformResult 转换结果
        """
        input_rows = len(data)
        current_data = data
        step_results: List[Dict[str, Any]] = []

        for step in self._steps:
            if not step.enabled:
                continue

            transformer = self._registry.get(step.transformer_name)
            result = transformer.execute(current_data, step.config)
            current_data = result.data

            step_results.append(
                {
                    "step_name": step.name or step.transformer_name,
                    "transformer": step.transformer_name,
                    "input_rows": result.input_rows,
                    "output_rows": result.output_rows,
                }
            )

        return TransformResult(
            data=current_data,
            input_rows=input_rows,
            output_rows=len(current_data),
            metadata={
                "chain_steps": len(self._steps),
                "step_results": step_results,
            },
        )

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        序列化为字典

        Returns:
            步骤列表的字典表示
        """
        return [
            {
                "transformer_name": step.transformer_name,
                "config": step.config,
                "name": step.name,
                "enabled": step.enabled,
            }
            for step in self._steps
        ]

    @classmethod
    def from_dict(
        cls,
        steps: List[Dict[str, Any]],
        registry: Optional["TransformerRegistry"] = None,
    ) -> "TransformChain":
        """
        从字典反序列化

        Args:
            steps: 步骤列表
            registry: 转换器注册中心

        Returns:
            TransformChain 实例
        """
        chain = cls(registry)
        for step in steps:
            chain.add(
                transformer_name=step["transformer_name"],
                config=step["config"],
                name=step.get("name", ""),
                enabled=step.get("enabled", True),
            )
        return chain

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> List[TransformStep]:
        """获取所有步骤"""
        return self._steps.copy()
