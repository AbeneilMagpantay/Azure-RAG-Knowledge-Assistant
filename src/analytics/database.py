"""Data store for loading and querying uploaded business data."""

import pandas as pd
import sqlite3
from typing import Optional, List, Dict, Any
from io import StringIO


class DataStore:
    """Load uploaded CSV data into SQLite for SQL queries."""
    
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.tables: Dict[str, pd.DataFrame] = {}
    
    def load_csv(self, file_content: bytes, table_name: str = "data") -> Dict[str, Any]:
        """
        Load CSV content into SQLite table.
        
        Args:
            file_content: Raw bytes from uploaded file
            table_name: Name for the SQL table
            
        Returns:
            Dict with table info (columns, rows, sample)
        """
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    content = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            df = pd.read_csv(StringIO(content))
            df.to_sql(table_name, self.conn, if_exists="replace", index=False)
            self.tables[table_name] = df
            
            return {
                "success": True,
                "table_name": table_name,
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample": df.head(5).to_dict('records')
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        try:
            return pd.read_sql_query(sql, self.conn)
        except Exception as e:
            raise ValueError(f"SQL Error: {e}")
    
    def get_tables(self) -> List[str]:
        """Get list of available tables."""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [row[0] for row in cursor.fetchall()]
    
    def get_schema(self, table: str) -> List[Dict[str, str]]:
        """Get column info for a table."""
        cursor = self.conn.execute(f"PRAGMA table_info({table})")
        return [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
    
    def get_dataframe(self, table: str) -> Optional[pd.DataFrame]:
        """Get DataFrame for a table."""
        return self.tables.get(table)
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect column types for analytics.
        
        Returns:
            Dict mapping column names to semantic types:
            - 'date': datetime columns
            - 'numeric': int/float columns
            - 'category': string columns with few unique values
            - 'text': string columns with many unique values
        """
        types = {}
        for col in df.columns:
            dtype = df[col].dtype
            
            # Check for date
            if pd.api.types.is_datetime64_any_dtype(dtype):
                types[col] = 'date'
            elif df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').any():
                types[col] = 'date'
            # Check for numeric
            elif pd.api.types.is_numeric_dtype(dtype):
                types[col] = 'numeric'
            # Check for category vs text
            elif dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1:  # Less than 10% unique
                    types[col] = 'category'
                else:
                    types[col] = 'text'
            else:
                types[col] = 'text'
        
        return types
    
    def suggest_analytics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Suggest analytics based on detected column types.
        
        Returns:
            Dict with suggested columns for different analytics
        """
        col_types = self.detect_column_types(df)
        
        suggestions = {
            "date_columns": [c for c, t in col_types.items() if t == 'date'],
            "numeric_columns": [c for c, t in col_types.items() if t == 'numeric'],
            "category_columns": [c for c, t in col_types.items() if t == 'category'],
            "recommended_charts": []
        }
        
        # Suggest charts based on column types
        if suggestions["date_columns"] and suggestions["numeric_columns"]:
            suggestions["recommended_charts"].append({
                "type": "time_series",
                "x": suggestions["date_columns"][0],
                "y": suggestions["numeric_columns"][0],
                "description": "Trend over time"
            })
        
        if suggestions["category_columns"] and suggestions["numeric_columns"]:
            suggestions["recommended_charts"].append({
                "type": "bar",
                "x": suggestions["category_columns"][0],
                "y": suggestions["numeric_columns"][0],
                "description": "Comparison by category"
            })
        
        if len(suggestions["numeric_columns"]) >= 2:
            suggestions["recommended_charts"].append({
                "type": "scatter",
                "x": suggestions["numeric_columns"][0],
                "y": suggestions["numeric_columns"][1],
                "description": "Correlation analysis"
            })
        
        return suggestions
