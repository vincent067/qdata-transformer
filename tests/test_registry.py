"""
注册中心和核心类测试
"""

import pytest
import polars as pl

from qdata_transformer import (
    TransformerRegistry,
    BaseTransformer,
    TransformResult,
    TransformStep,
    TransformChain,
)
from qdata_transformer.exceptions import TransformerNotFoundError


class DummyTransformer(BaseTransformer):
    """用于测试的虚拟转换器"""
    name = "dummy_transformer"
    description = "测试用转换器"
    version = "0.1.0"

    def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
        return data


class TestTransformerRegistry:
    """注册中心测试类"""

    def setup_method(self) -> None:
        """测试前清空注册中心"""
        # 保存原有状态
        self._original_transformers = TransformerRegistry._transformers.copy()
        self._original_instances = TransformerRegistry._instances.copy()

    def teardown_method(self) -> None:
        """测试后恢复注册中心"""
        TransformerRegistry._transformers = self._original_transformers
        TransformerRegistry._instances = self._original_instances

    def test_list_transformers(self) -> None:
        """测试列出所有转换器"""
        transformers = TransformerRegistry.list_transformers()

        assert isinstance(transformers, list)
        assert "polars_field_mapping" in transformers
        assert "duckdb_aggregation" in transformers

    def test_get_transformer(self) -> None:
        """测试获取转换器"""
        transformer = TransformerRegistry.get("polars_field_mapping")

        assert transformer is not None
        assert transformer.name == "polars_field_mapping"

    def test_get_nonexistent_transformer(self) -> None:
        """测试获取不存在的转换器"""
        with pytest.raises(TransformerNotFoundError):
            TransformerRegistry.get("nonexistent_transformer")

    def test_has_transformer(self) -> None:
        """测试检查转换器是否存在"""
        assert TransformerRegistry.has("polars_field_mapping") is True
        assert TransformerRegistry.has("nonexistent") is False

    def test_register_decorator(self) -> None:
        """测试装饰器注册"""
        @TransformerRegistry.register("test_decorator")
        class TestTransformer(BaseTransformer):
            name = "test_decorator"

            def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
                return data

        assert TransformerRegistry.has("test_decorator")
        transformer = TransformerRegistry.get("test_decorator")
        assert transformer.name == "test_decorator"

    def test_register_transformer_programmatic(self) -> None:
        """测试编程式注册"""
        TransformerRegistry.register_transformer("programmatic_test", DummyTransformer)

        assert TransformerRegistry.has("programmatic_test")
        transformer = TransformerRegistry.get("programmatic_test")
        assert isinstance(transformer, DummyTransformer)

    def test_unregister_transformer(self) -> None:
        """测试注销转换器"""
        TransformerRegistry.register_transformer("to_unregister", DummyTransformer)
        assert TransformerRegistry.has("to_unregister")

        result = TransformerRegistry.unregister("to_unregister")

        assert result is True
        assert TransformerRegistry.has("to_unregister") is False

    def test_unregister_nonexistent(self) -> None:
        """测试注销不存在的转换器"""
        result = TransformerRegistry.unregister("nonexistent_to_unregister")
        assert result is False

    def test_get_class(self) -> None:
        """测试获取转换器类"""
        cls = TransformerRegistry.get_class("polars_field_mapping")

        assert cls is not None
        assert cls.name == "polars_field_mapping"

    def test_create_instance(self) -> None:
        """测试创建新实例"""
        instance1 = TransformerRegistry.create_instance("polars_field_mapping")
        instance2 = TransformerRegistry.create_instance("polars_field_mapping")

        # create_instance 应该每次返回新实例
        assert instance1 is not instance2

    def test_get_singleton(self) -> None:
        """测试单例模式"""
        instance1 = TransformerRegistry.get("polars_field_mapping")
        instance2 = TransformerRegistry.get("polars_field_mapping")

        # get 应该返回相同实例
        assert instance1 is instance2

    def test_get_all_info(self) -> None:
        """测试获取所有转换器信息"""
        info = TransformerRegistry.get_all_info()

        assert isinstance(info, list)
        assert len(info) > 0

        for item in info:
            assert "name" in item
            assert "class_name" in item
            assert "description" in item
            assert "version" in item


class TestTransformResult:
    """转换结果测试类"""

    def test_create_result(self) -> None:
        """测试创建结果"""
        data = pl.DataFrame({"a": [1, 2, 3]})
        result = TransformResult(
            data=data,
            input_rows=5,
            output_rows=3,
            metadata={"transformer": "test"}
        )

        assert len(result.data) == 3
        assert result.input_rows == 5
        assert result.output_rows == 3
        assert result.metadata["transformer"] == "test"

    def test_rows_changed(self) -> None:
        """测试行数变化计算"""
        data = pl.DataFrame({"a": [1, 2, 3]})
        result = TransformResult(
            data=data,
            input_rows=5,
            output_rows=3
        )

        assert result.rows_changed == -2


class TestTransformStep:
    """转换步骤测试类"""

    def test_create_step(self) -> None:
        """测试创建步骤"""
        step = TransformStep(
            transformer_name="polars_field_mapping",
            config={"mappings": []},
            name="test_step",
            enabled=True
        )

        assert step.transformer_name == "polars_field_mapping"
        assert step.config == {"mappings": []}
        assert step.name == "test_step"
        assert step.enabled is True

    def test_step_defaults(self) -> None:
        """测试步骤默认值"""
        step = TransformStep(
            transformer_name="test",
            config={}
        )

        assert step.name == ""
        assert step.enabled is True


class TestBaseTransformer:
    """基础转换器测试类"""

    def test_get_info(self) -> None:
        """测试获取转换器信息"""
        transformer = DummyTransformer()
        info = transformer.get_info()

        assert info["name"] == "dummy_transformer"
        assert info["description"] == "测试用转换器"
        assert info["version"] == "0.1.0"

    def test_execute_returns_result(self) -> None:
        """测试执行返回结果"""
        transformer = DummyTransformer()
        data = pl.DataFrame({"a": [1, 2, 3]})

        result = transformer.execute(data, {})

        assert isinstance(result, TransformResult)
        assert result.input_rows == 3
        assert result.output_rows == 3
        assert result.metadata["transformer"] == "dummy_transformer"

    def test_pre_post_transform(self) -> None:
        """测试前后处理钩子"""
        class HookTransformer(BaseTransformer):
            name = "hook_transformer"
            pre_called = False
            post_called = False

            def transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
                return data

            def pre_transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
                HookTransformer.pre_called = True
                return data

            def post_transform(self, data: pl.DataFrame, config: dict) -> pl.DataFrame:
                HookTransformer.post_called = True
                return data

        transformer = HookTransformer()
        data = pl.DataFrame({"a": [1]})
        transformer.execute(data, {})

        assert HookTransformer.pre_called is True
        assert HookTransformer.post_called is True
