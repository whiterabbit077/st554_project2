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