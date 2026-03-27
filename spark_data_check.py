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
    
    
    # Return min and max summaries for one numeric column or all numeric columns
    def summarize_numeric_min_max(self, column_name: str = None, group_by: str = None):
        # Check that the grouping column exists if one was provided
        if group_by is not None and not self._column_exists(group_by):
            print(f"Column '{group_by}' was not found.")
            return None

        # If one column is supplied, check that it exists and is numeric
        if column_name is not None:
            if not self._column_exists(column_name):
                print(f"Column '{column_name}' was not found.")
                return None

            dtype = self._get_dtype(column_name)
            if not self._is_numeric_dtype(dtype):
                print(f"Column '{column_name}' is not numeric.")
                return None

            # Escape column names with special characters
            col_expr = F.col(f"`{column_name}`")

            # Return min and max without grouping
            if group_by is None:
                result = self.df.agg(
                    F.min(col_expr).alias(f"{column_name}_min"),
                    F.max(col_expr).alias(f"{column_name}_max")
                )
            # Return min and max with grouping
            else:
                result = self.df.groupBy(group_by).agg(
                    F.min(col_expr).alias(f"{column_name}_min"),
                    F.max(col_expr).alias(f"{column_name}_max")
                ).orderBy(group_by)

            return result.toPandas()

        # If no column is supplied, find all numeric columns
        numeric_columns = [
            col_name for col_name, dtype in self.df.dtypes
            if self._is_numeric_dtype(dtype)
        ]

        if not numeric_columns:
            return None

        # Return one-row summary for all numeric columns
        if group_by is None:
            agg_exprs = []
            for col_name in numeric_columns:
                col_expr = F.col(f"`{col_name}`")
                agg_exprs.append(F.min(col_expr).alias(f"{col_name}_min"))
                agg_exprs.append(F.max(col_expr).alias(f"{col_name}_max"))

            return self.df.agg(*agg_exprs).toPandas()

        # Return grouped summaries for all numeric columns
        grouped_results = []
        for col_name in numeric_columns:
            col_expr = F.col(f"`{col_name}`")
            one_result = self.df.groupBy(group_by).agg(
                F.min(col_expr).alias(f"{col_name}_min"),
                F.max(col_expr).alias(f"{col_name}_max")
            ).orderBy(group_by).toPandas()

            grouped_results.append(one_result)

        merged = reduce(
            lambda left, right: pd.merge(left, right, on=group_by, how="outer"),
            grouped_results
        )

        return merged

    # Return counts for one string column or two string columns
    def count_string_levels(self, column1: str, column2: str = None):
        # Check that the first column exists
        if not self._column_exists(column1):
            print(f"Column '{column1}' was not found.")
            return None

        # Check that the first column is a string
        dtype1 = self._get_dtype(column1)
        if not self._is_string_dtype(dtype1):
            print(f"Column '{column1}' is numeric.")
            return None

        # Count levels for one string column
        if column2 is None:
            result = self.df.groupBy(column1).count().orderBy(column1)
            return result.toPandas()

        # Check that the second column exists
        if not self._column_exists(column2):
            print(f"Column '{column2}' was not found.")
            return None

        # Check that the second column is a string
        dtype2 = self._get_dtype(column2)
        if not self._is_string_dtype(dtype2):
            print(f"Column '{column2}' is numeric.")
            return None

        # Count combinations for two string columns
        result = self.df.groupBy(column1, column2).count().orderBy(column1, column2)
        return result.toPandas()