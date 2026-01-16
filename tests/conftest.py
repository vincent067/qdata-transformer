"""
Pytest 配置文件
"""

import pytest
import polars as pl


@pytest.fixture
def sample_orders_data() -> pl.DataFrame:
    """订单数据样例"""
    return pl.DataFrame({
        "order_id": ["O001", "O002", "O003", "O004", "O005"],
        "customer_id": ["C001", "C002", "C001", "C003", "C002"],
        "product_id": ["P001", "P002", "P003", "P001", "P002"],
        "quantity": [2, 3, 1, 5, 2],
        "price": [100.0, 200.0, 150.0, 100.0, 200.0],
        "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "status": ["completed", "pending", "completed", "cancelled", "completed"],
    })


@pytest.fixture
def sample_user_data() -> pl.DataFrame:
    """用户数据样例"""
    return pl.DataFrame({
        "user_id": ["U001", "U002", "U003"],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
    })


@pytest.fixture
def sample_nested_data() -> pl.DataFrame:
    """嵌套数据样例"""
    return pl.DataFrame({
        "id": [1, 2, 3],
        "customer": [
            {"name": "Alice", "email": "alice@example.com"},
            {"name": "Bob", "email": "bob@example.com"},
            {"name": "Charlie", "email": "charlie@example.com"},
        ],
        "items": [
            [{"product": "A", "qty": 1}, {"product": "B", "qty": 2}],
            [{"product": "C", "qty": 3}],
            [{"product": "A", "qty": 1}],
        ],
    })
