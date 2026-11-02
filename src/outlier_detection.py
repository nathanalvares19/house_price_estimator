# to detect outliers for a feature
def detect_out(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    flags = (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)

    return flags
