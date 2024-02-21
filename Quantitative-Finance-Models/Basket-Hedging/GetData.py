import yfinance as yf
import pandas as pd


# Step 1: Get the NIKKEI composition
def get_nikkei_composition():
    # NIKKEI symbol
    components  = [
    "4151.T", "4502.T", "4503.T", "4506.T", "4507.T", "4519.T", "4523.T", "4568.T", "4578.T",
    "6479.T", "6501.T", "6503.T", "6504.T", "6506.T", "6594.T", "6645.T", "6674.T", "6701.T", "6702.T",
    "6723.T", "6724.T", "6752.T", "6753.T", "6758.T", "6762.T", "6770.T", "6841.T", "6857.T", "6861.T",
    "6902.T", "6920.T", "6952.T", "6954.T", "6971.T", "6976.T", "6981.T", "7735.T", "7751.T", "7752.T",
    "8035.T", "7201.T", "7202.T", "7203.T", "7205.T", "7211.T", "7261.T", "7267.T", "7269.T", "7270.T",
    "7272.T", "4543.T", "4902.T", "7731.T", "7733.T", "7741.T", "7762.T", "9432.T", "9433.T", "9434.T",
    "9613.T", "9984.T", "5831.T", "7186.T", "8304.T", "8306.T", "8308.T", "8309.T", "8316.T", "8331.T",
    "8354.T", "8411.T", "8253.T", "8591.T", "8697.T", "8601.T", "8604.T", "8630.T", "8725.T", "8750.T",
    "8766.T", "8795.T", "1332.T", "2002.T", "2269.T", "2282.T", "2501.T", "2502.T", "2503.T", "2531.T",
    "2801.T", "2802.T", "2871.T", "2914.T", "3086.T", "3099.T", "3382.T", "8233.T", "8252.T", "8267.T",
    "9843.T", "9983.T", "2413.T", "2432.T", "3659.T", "4324.T", "4385.T", "4661.T", "4689.T", "4704.T",
    "4751.T", "4755.T", "6098.T", "6178.T", "7974.T", "9602.T", "9735.T", "9766.T", "1605.T", "3401.T",
    "3402.T", "3861.T", "3863.T", "3405.T", "3407.T", "4004.T", "4005.T", "4021.T", "4042.T", "4043.T",
    "4061.T", "4063.T", "4183.T", "4188.T", "4208.T", "4452.T", "4631.T", "4901.T", "4911.T", "6988.T",
    "5019.T", "5020.T", "5101.T", "5108.T", "5201.T", "5214.T", "5232.T", "5233.T", "5301.T", "5332.T",
    "5333.T", "5401.T", "5406.T", "5411.T", "5541.T", "3436.T", "5706.T", "5711.T", "5713.T", "5714.T",
    "5801.T", "5802.T", "5803.T", "6201.T", "2768.T", "8001.T", "8002.T", "8015.T", "8031.T", "8053.T",
    "8058.T", "5631.T", "6103.T", "6113.T", "6273.T", "6301.T", "6302.T", "6305.T", "6326.T", "6361.T",
    "6367.T", "6471.T", "6472.T", "6473.T", "7004.T", "7011.T", "7013.T", "7012.T", "7832.T", "7911.T",
    "7912.T", "7951.T", "3289.T", "8801.T", "8802.T", "8804.T", "8830.T", "9001.T", "9005.T", "9007.T",
    "9008.T", "9009.T", "9020.T", "9021.T", "9022.T", "9064.T", "9147.T", "9101.T", "9104.T", "9107.T",
    "9201.T", "9202.T", "9301.T", "9501.T", "9502.T", "5101.T", "9108.T", "5201.T", "5214.T", "5232.T",
    "5233.T", "5301.T", "5332.T", "5333.T", "5401.T", "5406.T", "5411.T", "5541.T", "3436.T", "5706.T",
    "5711.T", "5713.T", "5714.T", "5801.T", "5802.T", "5803.T", "2768.T", "8001.T", "8002.T", "8015.T",
    "8031.T", "8053.T", "8058.T", "5631.T", "6103.T", "6113.T", "6273.T", "6301.T", "6302.T", "6305.T",
    "6326.T", "6361.T", "6367.T", "6471.T", "6472.T", "6473.T", "7004.T", "7011.T", "7013.T", "7012.T",
    "7832.T", "7911.T", "7912.T", "7951.T", "3289.T", "8801.T", "8802.T", "8804.T", "8830.T", "9001.T",
    "9005.T", "9007.T", "9008.T", "9009.T", "9020.T", "9021.T", "9022.T", "9064.T", "9147.T", "9101.T",
    "9104.T", "9107.T", "9201.T", "9202.T", "9301.T", "9501.T", "9502.T"
]

    return components


# Step 2: Get historical data for each stock in the NIKKEI index
def get_historical_data(components, start_date, end_date):
    stock_data = {}
    for symbol in components:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period='5y', start=start_date, end=end_date)
            if not data.empty:
                stock_data[symbol] = data['Close']
        except Exception as e:
            print(f"Error getting data for {symbol}: {str(e)}")

    return stock_data


# Step 3: Store the close prices in a CSV file
def save_to_csv(stock_data, output_file):
    df = pd.DataFrame(stock_data)
    df.to_csv(output_file)


if __name__ == '__main__':
    start_date = '2018-10-12'  # 5 years ago from the current date
    end_date = '2023-10-12'  # Current date

    components = get_nikkei_composition()
    stock_data = get_historical_data(components, start_date, end_date)

    if stock_data:
        save_to_csv(stock_data, 'nikkei_stock_data.csv')
        print("Data saved to 'nikkei_stock_data.csv'")
    else:
        print("No data retrieved.")
