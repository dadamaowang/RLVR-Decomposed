# -*- encoding: utf-8 -*-
'''
@File    :   get_mini_data.py
@Time    :   2025/11/29 22:02:25
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   None
@Desc    :   None
'''

import pandas as pd


def read_parquet(file_path: str):
    """Read a .parquet file into a pandas DataFrame."""
    df = pd.read_parquet(file_path)
    return df



if __name__ == "__main__":
    file_path = "/Users/katrina/Desktop/rlvr_decom/RLVR-Decomposed/data/amc23/test.parquet"

    """
    path:
    "/Users/katrina/Desktop/rlvr_decom/RLVR-Decomposed/data/math/train.parquet"
    "/Users/katrina/Desktop/rlvr_decom/RLVR-Decomposed/data/math/test.parquet"
    "/Users/katrina/Desktop/rlvr_decom/RLVR-Decomposed/data/amc23/test.parquet"
    "/Users/katrina/Desktop/rlvr_decom/RLVR-Decomposed/data/aime2025/test.parquet"
    
    """


    df = read_parquet(file_path)
    # print(df)
    sampled_df = df.sample(n=5, random_state=1127)

    sampled_df.to_parquet(
        '/Users/katrina/Desktop/rlvr_decom/RLVR-Decomposed/mini_data/amc23/test.parquet', 
        index=False)

    print(sampled_df)

    

