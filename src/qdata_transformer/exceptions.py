"""
数据转换引擎异常定义

提供转换引擎相关的所有异常类。
"""

from typing import Any, Dict, List, Optional


class TransformerError(Exception):
    """转换器基础异常"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class TransformerNotFoundError(TransformerError):
    """转换器未找到异常"""

    def __init__(self, transformer_name: str):
        self.transformer_name = transformer_name
        super().__init__(
            f"转换器未找到: {transformer_name}",
            {"transformer_name": transformer_name},
        )


class TransformerConfigError(TransformerError):
    """转换器配置错误"""

    def __init__(self, message: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"转换器配置错误: {message}",
            {"config": config} if config else {},
        )


class TransformExecutionError(TransformerError):
    """转换执行错误"""

    def __init__(
        self,
        message: str,
        transformer_name: str,
        original_error: Optional[Exception] = None,
    ):
        self.transformer_name = transformer_name
        self.original_error = original_error
        details: Dict[str, Any] = {"transformer_name": transformer_name}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(
            f"转换执行错误 [{transformer_name}]: {message}",
            details,
        )


class MappingConfigError(TransformerConfigError):
    """映射配置错误"""

    def __init__(self, message: str, field_name: Optional[str] = None):
        details: Dict[str, Any] = {}
        if field_name:
            details["field_name"] = field_name
        super().__init__(message, details)


class AggregationConfigError(TransformerConfigError):
    """聚合配置错误"""

    pass


class InvalidColumnError(TransformerError):
    """无效列错误"""

    def __init__(self, column_name: str, available_columns: Optional[List[str]] = None):
        self.column_name = column_name
        details: Dict[str, Any] = {"column_name": column_name}
        if available_columns:
            details["available_columns"] = available_columns
        super().__init__(
            f"无效列名: {column_name}",
            details,
        )
