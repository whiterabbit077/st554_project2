from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql.types import *
import pandas as pd


class SparkDataCheck:
    def __init__(self, dataframe: DataFrame):
        self.df = dataframe

    @classmethod
    def from_csv(cls, spark, path: str):
        df = spark.read.load(
            path,
            format="csv",
            sep=",",
            inferSchema=True,
            header=True
        )
        return cls(df)

    @classmethod
    def from_pandas(cls, spark, pandas_df: pd.DataFrame):
        df = spark.createDataFrame(pandas_df)
        return cls(df)

    def _column_exists(self, column_name: str) -> bool:
        return column_name in self.df.columns

    def _get_dtype(self, column_name: str):
        dtype_dict = dict(self.df.dtypes)
        return dtype_dict.get(column_name)

    def _is_numeric_dtype(self, dtype: str) -> bool:
        numeric_types = {
            "int", "bigint", "double", "float",
            "long", "smallint", "tinyint", "decimal"
        }
        if dtype is None:
            return False
        return any(dtype.startswith(x) for x in numeric_types)

    def _is_string_dtype(self, dtype: str) -> bool:
        return dtype == "string"
    
    
    # Check whether values in a numeric column fall within the given range
    def check_numeric_range(self, column_name: str, lower=None, upper=None, new_col_name: str = None):
        # Check that the column exists
        if not self._column_exists(column_name):
            print(f"Column '{column_name}' was not found.")
            return self

        # Check that the column is numeric
        dtype = self._get_dtype(column_name)
        if not self._is_numeric_dtype(dtype):
            print(f"Column '{column_name}' is not numeric.")
            return self

        # Make sure at least one bound is provided
        if lower is None and upper is None:
            print("Provide at least one of lower or upper.")
            return self

        # Create a default name for the new Boolean column if needed
        if new_col_name is None:
            new_col_name = f"{column_name}_in_range"

        col = F.col(column_name)

        # Build the logical condition depending on which bounds were given
        if lower is not None and upper is not None:
            condition = col.between(lower, upper)
        elif lower is not None:
            condition = col >= lower
        else:
            condition = col <= upper

        # Append the new Boolean column, keeping NULL values as NULL
        self.df = self.df.withColumn(
            new_col_name,
            F.when(col.isNull(), F.lit(None)).otherwise(condition)
        )

        return self

    # Check whether values in a string column belong to a set of valid levels
    def check_string_levels(self, column_name: str, levels, new_col_name: str = None):
        # Check that the column exists
        if not self._column_exists(column_name):
            print(f"Column '{column_name}' was not found.")
            return self

        # Check that the column is a string
        dtype = self._get_dtype(column_name)
        if not self._is_string_dtype(dtype):
            print(f"Column '{column_name}' is not a string column.")
            return self

        # Create a default name for the new Boolean column if needed
        if new_col_name is None:
            new_col_name = f"{column_name}_valid_level"

        col = F.col(column_name)

        # Append the new Boolean column, keeping NULL values as NULL
        self.df = self.df.withColumn(
            new_col_name,
            F.when(col.isNull(), F.lit(None)).otherwise(col.isin(levels))
        )

        return self

    # Check whether values in a column are missing
    def check_missing(self, column_name: str, new_col_name: str = None):
        # Check that the column exists
        if not self._column_exists(column_name):
            print(f"Column '{column_name}' was not found.")
            return self

        # Create a default name for the new Boolean column if needed
        if new_col_name is None:
            new_col_name = f"{column_name}_is_missing"

        # Append the new Boolean column showing whether each value is NULL
        self.df = self.df.withColumn(
            new_col_name,
            F.col(column_name).isNull()
        )

        return self