"""
转换器注册中心

提供转换器的注册、获取和管理功能。
支持装饰器注册和编程式注册。
"""

from typing import Any, Callable, Dict, List, Optional, Type

from qdata_transformer.base import BaseTransformer
from qdata_transformer.exceptions import TransformerNotFoundError


class TransformerRegistry:
    """
    转换器注册中心

    管理所有注册的转换器，提供统一的访问接口。

    使用示例：
        # 装饰器注册
        @TransformerRegistry.register()
        class MyTransformer(BaseTransformer):
            name = "my_transformer"
            ...

        # 指定名称注册
        @TransformerRegistry.register("custom_name")
        class AnotherTransformer(BaseTransformer):
            ...

        # 编程式注册
        TransformerRegistry.register_transformer("my_transformer", MyTransformer)

        # 获取转换器
        transformer = TransformerRegistry.get("my_transformer")
    """

    # 全局转换器存储
    _transformers: Dict[str, Type[BaseTransformer]] = {}
    _instances: Dict[str, BaseTransformer] = {}

    def __init__(self) -> None:
        """初始化注册中心"""
        pass

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
    ) -> Callable[[Type[BaseTransformer]], Type[BaseTransformer]]:
        """
        注册转换器装饰器

        Args:
            name: 转换器名称（可选，默认使用类的 name 属性）

        Returns:
            装饰器函数

        使用示例：
            @TransformerRegistry.register()
            class MyTransformer(BaseTransformer):
                name = "my_transformer"

            @TransformerRegistry.register("custom_name")
            class AnotherTransformer(BaseTransformer):
                name = "another"  # 会被 "custom_name" 覆盖
        """

        def decorator(transformer_class: Type[BaseTransformer]) -> Type[BaseTransformer]:
            key = name or transformer_class.name
            cls._transformers[key] = transformer_class
            return transformer_class

        return decorator

    @classmethod
    def register_transformer(
        cls,
        name: str,
        transformer_class: Type[BaseTransformer],
    ) -> None:
        """
        编程式注册转换器

        Args:
            name: 转换器名称
            transformer_class: 转换器类
        """
        cls._transformers[name] = transformer_class

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        注销转换器

        Args:
            name: 转换器名称

        Returns:
            是否成功注销
        """
        if name in cls._transformers:
            del cls._transformers[name]
            if name in cls._instances:
                del cls._instances[name]
            return True
        return False

    @classmethod
    def get(cls, name: str) -> BaseTransformer:
        """
        获取转换器实例

        使用单例模式，每个转换器只创建一个实例。

        Args:
            name: 转换器名称

        Returns:
            转换器实例

        Raises:
            TransformerNotFoundError: 转换器未找到时抛出
        """
        if name not in cls._transformers:
            raise TransformerNotFoundError(name)

        # 单例模式
        if name not in cls._instances:
            cls._instances[name] = cls._transformers[name]()

        return cls._instances[name]

    @classmethod
    def get_class(cls, name: str) -> Type[BaseTransformer]:
        """
        获取转换器类

        Args:
            name: 转换器名称

        Returns:
            转换器类

        Raises:
            TransformerNotFoundError: 转换器未找到时抛出
        """
        if name not in cls._transformers:
            raise TransformerNotFoundError(name)
        return cls._transformers[name]

    @classmethod
    def create_instance(cls, name: str) -> BaseTransformer:
        """
        创建新的转换器实例

        与 get() 不同，此方法总是创建新实例。

        Args:
            name: 转换器名称

        Returns:
            新的转换器实例

        Raises:
            TransformerNotFoundError: 转换器未找到时抛出
        """
        if name not in cls._transformers:
            raise TransformerNotFoundError(name)
        return cls._transformers[name]()

    @classmethod
    def list_transformers(cls) -> List[str]:
        """
        列出所有注册的转换器名称

        Returns:
            转换器名称列表
        """
        return list(cls._transformers.keys())

    @classmethod
    def get_all_info(cls) -> List[Dict[str, Any]]:
        """
        获取所有转换器的信息

        Returns:
            转换器信息列表
        """
        result = []
        for name, transformer_class in cls._transformers.items():
            result.append(
                {
                    "name": name,
                    "class_name": transformer_class.__name__,
                    "description": getattr(transformer_class, "description", ""),
                    "version": getattr(transformer_class, "version", "1.0.0"),
                }
            )
        return result

    @classmethod
    def has(cls, name: str) -> bool:
        """
        检查转换器是否存在

        Args:
            name: 转换器名称

        Returns:
            是否存在
        """
        return name in cls._transformers

    @classmethod
    def clear(cls) -> None:
        """
        清空所有注册的转换器

        主要用于测试
        """
        cls._transformers.clear()
        cls._instances.clear()
