from typing_extensions import runtime
from __future__ import annotations 
from typing import Any


from dataclasses import dataclass 

import numpy as np 
import pandas as pd 

@dataclass 
class SplitResult: 
    train: pd.DataFrame 
    validation: pd.DataFrame
    testL pd.DataFrame 


class ProductSplitter: 
    def __init__(
        selfm 
        config: Any | None = None, 
        *, 
        product_col: str = "product_id", 
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> None: 

        self.config = config 

        self.product_col = self._get_config_value(
            "product_col", product_col,
        )

        self.train_size = float(
            self._get_config_value(
                "train_size", train_size,
            )
        )

        self.val_size = float(
            self._get_config_value(
                "val_size", val_size,
            )
        )

        self.test_size = float(
            self._get_config_value(
                "test_size", test_size,
            )
        )

        self.random_state = int(
            self._get_config_value(
                "random_state", random_state,
            )
        )

        self._validate_split_sizes()


    def split(
        self, df: pd.DataFrame, 
    ) -> SplitResult: 

        self._validate_dataframe(df)

        data = df.copy()

        products = (
            data[self.product_col]
            .dropna()
            .drop_duplicates()
            .to_numpy()
        )

        if len(products) < 3: 
            raise ValueError(f"Dibutuhkan setidaknya 3 produk, tersedia {len(products)}.")
        
        rng = np.random.default_rng(self.random_state)

        products = rng.permutation(products)

        (
            train_products, val_products, test_products,
        ) = self._split_products(products)

        train_df = (
            data[
                data[self.product_col].isin(train_products)
            ].copy().reset_index(drop = True)
        )

        val_df = (
            data[
                data[self.product_col].isin(val_products)
            ].copy().reset_index(drop = True)
        )

        test_df = (
            data[
                data[self.product_col].isin(test_products)
            ].copy().reset_index(drop = True)   
        )

        self._validate_no_leakage(
            train_df, val_df, test_df,
        )

        return SplitResult(
            train=train_df, validation=val_df, test=test_df,
        )

    
    def summary(
        self, split_result: SplitResult,
    ) -> pd.DataFrame:
        rows = []

        datasets = {
            "train": split_result.train, 
            "validation": split_result.validation, 
            "test":split_result.test,
        }

        total_rows = sum(
            len(dataset) 
            for dataset in datasets.values()
        )
        
        total_products = sum(
            dataset[self.product_col].nunique()
            for dataset in datasets.values()
        )

        for name, dataset in datasets.items():
            rows.append(
                {
                    "split": name, 
                    "rows": len(dataset), 
                    "row_ratio": (
                        len(dataset) / total_rows
                        if total_rows
                        else 0
                    ), 
                    "products": dataset[
                        self.product_col
                    ].nunique(), 
                    "product_ratio": (
                        dataset[self.product_col]
                        .nunique() / total_products
                        if total_products
                        else 0
                    ),
                    
                }
            )
        return pd.DataFrame(rows)

    
    def _split_products(
        self, products: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: 
        
        n_product = len(products)

        n_train = int(
            np.floor(
                n_product * self.train_size 
            )
        )
        

        n_val = int(
            np.floor(
                n_product * self.val_size
            )
        )
        n_test = n_product - n_train - n_val

        if n_train == 0: 
            raise ValueError(
                "train split menghasilkan 0 produk"
            )

        if n_val == 0:
            raise ValueError(
                "val split menghasilkan 0 produk"
            )
        
        if n_test == 0:
            raise ValueError(
                "test split menghasilkan 0 produk"
            )
        
        train_end = n_train 
        val_end = train_end + n_val 

        train_products = products[:train_end]
        
        val_products = products[train_end:val_end]

        test_products = products[val_end:]

        return (
            train_products, val_products, test_products,
        )
        
    def _validate_dataframe(
        self, df:pd.DataFrame,
    ) -> None: 

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Input harus pd.DataFrame, tapi didapat {type(df).__name__}",
            )

        if df.empty:
            raise ValueError(
                "DataFrame kosong."
            )

        if self.product_col not in df.columns:
            raise KeyError(
                f"Kolom '{self.product_col}' tidak ditemukan"
            )

        missing_products = (
            df[self.product_col]
            .isna()
            .mean()
        )

        if missing_products > 0:
            raise ValueError(
                f"ditemukan {missing_products} baris"
                f"dengan {self.product_col} null"
            )

        
    def _validate_split_sizes(self) -> None:
        sizes = [
            self.train_size, 
            self.val_size, 
            self.test_size,
        ]

        if any(
            size <= 0
            for size in sizes
        ): 
            raise ValueError(
                f"ukuran split harus >0, didapat {sizes}"
            )
        total = sum(sizes)

        if not np.isclose(total, 1.0):
            raise ValueError(
                f"jumlah ukuran split harus 1.0, didapat {total}"
            )
        
    
    def _validate_no_leakage(
        self, train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
    ) -> None: 

        train_products = set(
            train_df[self.product_col].unique()
        )

        val_products = set(
            val_df[self.product_col].unique()
        )

        test_products = set(
            test_df[self.product_col].unique()
        )

        train_val_overlap = (
            train_products & val_products
        )

        train_test_overlap = (
            train_products & test_products
        )

        val_test_overlap = (
            val_products & test_products
        )

        if train_val_overlap or train_test_overlap or val_test_overlap: 
            raise RuntimeError(
                "Data leakage terdeteksi"
            )

        

    def _get_config_value(
        self, name: str, defaults: Any
    ) -> Any: 
        if self.config is None:
            return defaults 
        return getattr(
            self.config, name, defaults,
        )