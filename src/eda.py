# inspect data
def inspect(data):
    print(f"Shape: {data.shape}\n")
    # print(f"Info:")
    # data.info()
    # print("\n")
    print(f"Null value count:\n{data.isna().sum()}\n")
    print(f"Duplicate rows:\n{data.duplicated().sum()}\n")
