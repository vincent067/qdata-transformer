"""
数据拆分和去重转换器测试
"""

import pytest
import polars as pl

from qdata_transformer import (
    PolarsSplitTransformer,
    PolarsDeduplicateTransformer,
    PolarsMergeTransformer,
)
from qdata_transformer.exceptions import TransformerConfigError, TransformExecutionError


class TestPolarsSplitTransformer:
    """数据拆分转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = PolarsSplitTransformer()

    def test_explode_list(self) -> None:
        """测试列表展开"""
        data = pl.DataFrame({
            "id": [1, 2],
            "items": [[1, 2, 3], [4, 5]]
        })

        config = {
            "column": "items",
            "type": "explode"
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 5  # 3 + 2 = 5 行
        assert result.data["items"].to_list() == [1, 2, 3, 4, 5]

    def test_split_string(self) -> None:
        """测试字符串拆分"""
        data = pl.DataFrame({
            "id": [1, 2],
            "tags": ["a,b,c", "d,e"]
        })

        config = {
            "column": "tags",
            "type": "split",
            "separator": ","
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 5  # 3 + 2 = 5 行
        assert result.data["tags"].to_list() == ["a", "b", "c", "d", "e"]

    def test_unnest_struct(self) -> None:
        """测试结构体展开"""
        data = pl.DataFrame({
            "id": [1, 2],
            "info": [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
        })

        config = {
            "column": "info",
            "type": "unnest"
        }

        result = self.transformer.execute(data, config)

        assert "name" in result.data.columns
        assert "age" in result.data.columns
        assert result.data["name"].to_list() == ["Alice", "Bob"]

    def test_missing_column_config(self) -> None:
        """测试缺少 column 配置"""
        data = pl.DataFrame({"id": [1, 2]})

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(data, {"type": "explode"})

    def test_invalid_column(self) -> None:
        """测试无效列名"""
        data = pl.DataFrame({"id": [1, 2]})

        config = {
            "column": "nonexistent",
            "type": "explode"
        }

        with pytest.raises(TransformExecutionError):
            self.transformer.execute(data, config)

    def test_split_missing_separator(self) -> None:
        """测试 split 缺少 separator"""
        data = pl.DataFrame({"tags": ["a,b,c"]})

        config = {
            "column": "tags",
            "type": "split"
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(data, config)


class TestPolarsDeduplicateTransformer:
    """数据去重转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = PolarsDeduplicateTransformer()

    def test_simple_deduplicate(self) -> None:
        """测试简单去重"""
        data = pl.DataFrame({
            "id": [1, 2, 2, 3, 3, 3],
            "value": ["a", "b", "b", "c", "c", "c"]
        })

        config = {
            "columns": ["id"],
            "keep": "first"
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 3
        assert result.data["id"].to_list() == [1, 2, 3]

    def test_deduplicate_keep_last(self) -> None:
        """测试保留最后一个"""
        data = pl.DataFrame({
            "id": [1, 1, 2, 2],
            "value": ["a", "b", "c", "d"]
        })

        config = {
            "columns": ["id"],
            "keep": "last",
            "maintain_order": True
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 2
        # 保留 id=1 的最后一行（value="b"）和 id=2 的最后一行（value="d"）
        assert "b" in result.data["value"].to_list()
        assert "d" in result.data["value"].to_list()

    def test_deduplicate_multiple_columns(self) -> None:
        """测试多列去重"""
        data = pl.DataFrame({
            "id": [1, 1, 1, 2],
            "category": ["A", "A", "B", "A"],
            "value": [1, 2, 3, 4]
        })

        config = {
            "columns": ["id", "category"],
            "keep": "first"
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 3  # (1,A), (1,B), (2,A)

    def test_deduplicate_all_columns(self) -> None:
        """测试全列去重"""
        data = pl.DataFrame({
            "id": [1, 1, 2],
            "value": ["a", "a", "b"]
        })

        config = {
            "keep": "first"
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 2

    def test_invalid_keep_strategy(self) -> None:
        """测试无效的 keep 策略"""
        data = pl.DataFrame({"id": [1, 2]})

        config = {
            "columns": ["id"],
            "keep": "invalid"
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(data, config)

    def test_invalid_column(self) -> None:
        """测试无效列名"""
        data = pl.DataFrame({"id": [1, 2]})

        config = {
            "columns": ["nonexistent"],
            "keep": "first"
        }

        with pytest.raises(TransformExecutionError):
            self.transformer.execute(data, config)


class TestPolarsMergeTransformer:
    """数据合并转换器测试类"""

    def setup_method(self) -> None:
        """测试前初始化"""
        self.transformer = PolarsMergeTransformer()

    def test_rechunk(self) -> None:
        """测试 rechunk"""
        data = pl.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"]
        })

        config = {
            "rechunk": True
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 3

    def test_no_rechunk(self) -> None:
        """测试不 rechunk"""
        data = pl.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"]
        })

        config = {
            "rechunk": False
        }

        result = self.transformer.execute(data, config)

        assert len(result.data) == 3

    def test_invalid_how(self) -> None:
        """测试无效的 how 参数"""
        data = pl.DataFrame({"id": [1, 2]})

        config = {
            "how": "invalid"
        }

        with pytest.raises(TransformerConfigError):
            self.transformer.execute(data, config)
