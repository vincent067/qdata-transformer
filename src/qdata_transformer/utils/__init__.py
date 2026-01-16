"""
SQL 工具函数

提供 DuckDB SQL 转换器共享的工具函数。
"""

from typing import Any


def quote_identifier(identifier: str) -> str:
    """
    引用 SQL 标识符（处理特殊字符）

    如果标识符包含特殊字符或是保留字，使用双引号包裹。

    Args:
        identifier: SQL 标识符（列名、表名等）

    Returns:
        安全的 SQL 标识符

    示例：
        >>> quote_identifier("customer_id")
        'customer_id'
        >>> quote_identifier("column with space")
        '"column with space"'
        >>> quote_identifier("SELECT")
        '"SELECT"'
    """
    # SQL 保留字列表（常见的）
    reserved_words = {
        "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER",
        "LIMIT", "OFFSET", "JOIN", "ON", "AND", "OR", "NOT", "IN",
        "IS", "NULL", "AS", "DISTINCT", "COUNT", "SUM", "AVG", "MIN",
        "MAX", "BETWEEN", "LIKE", "CASE", "WHEN", "THEN", "ELSE", "END",
        "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TABLE",
        "INDEX", "VIEW", "UNION", "ALL", "ANY", "EXISTS", "TRUE", "FALSE",
    }

    if not identifier.isidentifier() or identifier.upper() in reserved_words:
        return f'"{identifier}"'
    return identifier


def format_sql_value(value: Any) -> str:
    """
    格式化 SQL 值

    将 Python 值转换为 SQL 字面量表示。

    Args:
        value: Python 值

    Returns:
        SQL 字面量字符串

    示例：
        >>> format_sql_value(None)
        'NULL'
        >>> format_sql_value("hello")
        "'hello'"
        >>> format_sql_value(42)
        '42'
    """
    if value is None:
        return "NULL"
    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    elif isinstance(value, str):
        # 转义单引号
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        # 其他类型转为字符串
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
