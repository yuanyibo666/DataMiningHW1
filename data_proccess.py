import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import time
import re
from datetime import datetime, timezone

def process_10G_files(file_paths='10G_data/part-0000', batch_size=1000000):
    start_time = time.time()
    total_rows = 0
    processed_rows = 0
    missing_counts = pd.Series(0, index=[
        'timestamp', 'user_name', 'chinese_name', 'email', 'age', 'income',
        'gender', 'country', 'chinese_address', 'purchase_history', 'is_active',
        'registration_date', 'credit_score', 'phone_number'
    ])
    outlier_counts = pd.Series(0, index=['age', 'income', 'credit_score'])
    duplicate_counts = 0
    deleted_rows = 0

    # 处理 16 个 Parquet 文件
    for i in range(16):
        if i < 10:
            file_path = f"{file_paths}{i}.parquet"
        else:
            file_path = f"10G_data/part-000{i}.parquet"
        
        parquet_file = pq.ParquetFile(file_path)
        file_rows = parquet_file.metadata.num_rows
        total_rows += file_rows
        print(f"file {file_path}: {file_rows} rows")

        # 分批处理
        batch_idx = 0
        for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size), desc=f"proccesing file {i}"):
            batch_time = time.time()
            batch_df = batch.to_pandas()

            # 1. 缺失值统计
            missing_counts += batch_df.isnull().sum()

            # 2. 重复值统计
            duplicate_counts += batch_df.duplicated().sum()

            # 3. 异常值检测
            # 年龄：负值或 > 120
            age_outliers = batch_df[(batch_df['age'] < 0) | (batch_df['age'] > 120)]
            outlier_counts['age'] += len(age_outliers)

            # 收入：负值或 > 10亿
            income_outliers = batch_df[(batch_df['income'] < 0) | (batch_df['income'] > 1e9)]
            outlier_counts['income'] += len(income_outliers)

            # 信用评分：< 300 或 > 850
            credit_outliers = batch_df[(batch_df['credit_score'] < 300) | (batch_df['credit_score'] > 850)]
            outlier_counts['credit_score'] += len(credit_outliers)

            # 4. 数据清洗
            original_len = len(batch_df)

            # 删除异常值（年龄、收入、信用评分）
            batch_df = batch_df[
                (batch_df['age'] >= 0) & (batch_df['age'] <= 120) &
                (batch_df['income'] >= 0) & (batch_df['income'] <= 1e9) &
                (batch_df['credit_score'] >= 300) & (batch_df['credit_score'] <= 850)
            ]

            # 删除重复行
            batch_df = batch_df.drop_duplicates()

            # 格式规范化
            # 转换为 datetime
            batch_df['timestamp'] = pd.to_datetime(batch_df['timestamp'], errors='coerce')
            batch_df['registration_date'] = pd.to_datetime(batch_df['registration_date'], errors='coerce')

            # 验证 email 格式
            def is_valid_email(email):
                pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                return bool(re.match(pattern, email)) if isinstance(email, str) else False
            invalid_emails = batch_df['email'].apply(lambda x: not is_valid_email(x))
            batch_df = batch_df[~invalid_emails]

            # 规范化 gender
            valid_genders = ['Male', 'Female', 'Other']
            batch_df['gender'] = batch_df['gender'].apply(lambda x: x if x in valid_genders else 'Other')

            # 确保 is_active 为布尔值
            batch_df['is_active'] = batch_df['is_active'].astype(bool)

            # 删除后的行数变化
            deleted_rows += original_len - len(batch_df)

            # 保存清洗后的批次
            batch_df.to_parquet(f"new_data/30G_cleaned_batch_{i}_{batch_idx}.parquet", index=False)
            processed_rows += len(batch_df)
            batch_idx += 1

            print(f"batch {batch_idx} (file {i}) time consume: {time.time() - batch_time:.2f} second")

    # 统计结果
    missing_ratio = missing_counts / total_rows
    outlier_ratio = outlier_counts / total_rows
    duplicate_ratio = duplicate_counts / total_rows

    print("\n=== data quality evaluation ===")
    print(f"total_rows: {total_rows}")
    print(f"processed_rows: {processed_rows}")
    print(f"deleted_rows: {deleted_rows}")
    print(f"missing_counts:\n{missing_counts}")
    print(f"missing_ratio:\n{missing_ratio}")
    print(f"outlier_counts:\n{outlier_counts}")
    print(f"outlier_ratio:\n{outlier_ratio}")
    print(f"duplicate_counts: {duplicate_counts}")
    print(f"duplicate_ratio: {duplicate_ratio}")
    print(f"processing time: {time.time() - start_time:.2f} seconds")

    return {
        'total_rows': total_rows,
        'processed_rows': processed_rows,
        'deleted_rows': deleted_rows,
        'missing_counts': missing_counts,
        'missing_ratio': missing_ratio,
        'outlier_counts': outlier_counts,
        'outlier_ratio': outlier_ratio,
        'duplicate_counts': duplicate_counts,
        'duplicate_ratio': duplicate_ratio
    }

# 示例调用
if __name__ == "__main__":
    result = process_10G_files(file_paths='30G_data/part-0000', batch_size=1000000)