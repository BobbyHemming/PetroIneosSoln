import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def main_question_1(plot=True):
    """
    -> You are provided with data with 10-mins, 60-mins and 1-day resolution (Merge.csv).
    -> Please merge them into a pandas Dataframe with 2-hours resolution in between 7:00 – 17:00 only as index.
    -> Please take the average of the 10-mins and 60-mins resolution prices during the 2-hours window and forward fill
       the 1-day resolution prices in the 2-hours window.
    """

    # 2 hour resolution -> look at period e.g. price at 9am is average of period 8-10am,

    # import merge data
    df = pd.read_csv('data/Merge.csv')
    df = df.sort_values('Datetime')
    df = df.set_index(pd.to_datetime(df['Datetime']))

    # Create empty dataframe with correct datetime as index, we use this to merge later
    date_range = pd.date_range(start=min(df['Datetime']), end=max(df['Datetime']), freq='2h')
    date_range = date_range + pd.DateOffset(hours=1)
    df_f = pd.DataFrame(index=date_range).between_time('7:00', '16:59')  # built our ideal datetime resolution

    def resolve_high_frequency_data(data):
        # function included to follow DNRY principles
        data['Price'] = data['Price'].ffill()  # cleaning *** ASSUMPTION: forward fill price where NAN ***
        data['Avg Price'] = data['Price'].shift(freq='-1h').rolling('2h').mean()  # rolling window of 2hr
        return data

    # Separate dataframes into separate resolutions
    # 10 mins
    df_10mins = df[df['Resolution'] == '10MIN'].copy()
    df_10mins = resolve_high_frequency_data(df_10mins)
    # 1 hour
    df_1h = df[df['Resolution'] == '1H'].copy()
    df_1h = resolve_high_frequency_data(df_1h)
    # 1 day
    df_1d = df[df['Resolution'] == 'D'].copy()
    df_1d = df_1d.set_index(pd.to_datetime(df_1d['Datetime'])+pd.DateOffset(hours=7))

    # Merge the different resolutions
    # CONCATINATION with correct axis would work (but relies on no gaps in data) -> safer to merge twice for 3 dfs
    df_10mins = pd.merge(df_f, df_10mins, how='inner', left_index=True, right_index=True).dropna(subset=['Avg Price'])
    df_1h_10min = pd.merge(df_10mins['Avg Price'], df_1h['Avg Price'], how='inner', left_index=True, right_index=True)
    # left merge between df_1d and the df_1h_10min as df_1d isn't resolved to 2 hourly periods yet
    df_combined = pd.merge(df_1h_10min, df_1d["Price"], how='left', left_index=True, right_index=True).ffill()
    df_combined.index.name = 'Datetime'   # remember column names: 10min-"Avg Price_x", 1h-"Avg Price_y", 1d-"Price".
    df_combined['avg_price'] = df_combined.apply(lambda x: np.mean(x), axis=1)  # final mean of all the prices at 2hr resolution

    if plot:
        plt.figure()
        plt.plot(df_combined.index, df_combined['Avg Price_x'], label='10min')
        plt.plot(df_combined.index, df_combined['Avg Price_y'], label='1hr')
        plt.plot(df_combined.index, df_combined['Price'], label='1day')
        plt.plot(df_combined.index, df_combined['avg_price'], label='mean', ls='--')
        plt.title('Q1: 10min, 60min, 1day Data Resolution')
        plt.legend()
        plt.show()

    print('\nQ1:')
    print('The output of function main_question_1() includes the dataframe with the required format as per the instructions.')
    print('A PLOT is also included (optional), to demonstrate the different resolutions of the data and the combined average.')
    return df_combined


def main_question_2():
    """
    -> You are provided with a daily energy consumption data from 2016 to date (Consumption.csv).
    -> Please create a Pandas DataFrame with to show the consumption of each year. The expected format is to have the
       year number as column name and mm-dd as index.
    -> Please also create a seasonal plot showing 5-years (2016-2020) range (shaded) & average (dashed line),
       and year 2021 (line) & 2022 (line).
    -> Please comment on your observation on the plot
    """
    df = pd.read_csv('data/Consumption.csv')

    def normalise_date(data):
        previous_date = data['Date'].iloc[0]
        for i, row in data.iterrows():
            date_fixed = validate_modify_date(row['Date'], previous_date)
            data.loc[i, 'Date'] = date_fixed
            previous_date = date_fixed

        return data

    def validate_modify_date(x, previous_date):
        """ Some of the dates provided in Consumption.csv are in the wrong format to be converted to Datetimes.
            This function will fix the strings so they are in the correct format where necessary"""
        previous_month = previous_date.split('/')[1]
        try:
            datetime.datetime.strptime(x, "%d/%m/%Y")
            return x
        except ValueError:
            yyyy = x[:4]
            if len(x[4:]) == 2:
                mm = '0' + x[4:5]
                dd = '0' + x[5:]
                # print(dd+'/'+mm+'/'+yyyy)
            elif len(x[4:]) == 3:
                # problem here is we have two dates exactly the same, e.g. 2020111 and 2020111
                # one means 2020 01 11 the other means 2020 11 01
                # we need to differentiate between them, the only way is by using the prior date
                if int(previous_month) == 9 and int(x[4]) == 1:
                    mm = x[4:6]
                    dd = '0' + x[6:]
                elif 9 <= int(previous_month) <= 12 and int(x[4]) == 1:
                    mm = x[4:6]
                    dd = '0' + x[6:]
                elif 1 <= int(previous_month) <= 9 and 1 <= int(x[4]) <= 9:
                    mm = '0' + x[4:5]
                    dd = x[5:]
                else:
                    raise ValueError('Unable to parse unrecognised date format: ', previous_month, x[4])
            else:
                mm = x[4:6]
                dd = x[6:]

            x = dd+'/'+mm+'/'+yyyy
            return x

    df = normalise_date(df)

    # Build correct "day/month", "year" columns
    df['dd/mm'] = df['Date'].apply(lambda x: '/'.join(str(x).split('/')[:2]))
    df['yyyy'] = df['Date'].apply(lambda x: str(x).split('/')[-1])
    df = df[['dd/mm', 'yyyy', 'Consumption']]

    # Importantly pivot the table so we have yyyy's as columns, dd/mm as rows (index)
    df = df.pivot(index='dd/mm', columns='yyyy', values='Consumption')

    # Some formating chnages to get the data looking how we wnat it to
    df['dd/mm'] = df.index
    df['mm/dd'] = df['dd/mm'].apply(lambda x: x.split('/')[1]+'-'+x.split('/')[0])
    df = df.sort_values('mm/dd')
    df = df.set_index(df['mm/dd'], drop=True)
    df = df.drop(columns=['dd/mm', 'mm/dd'])

    # Plot the results
    fig, ax = plt.subplots()

    years_range = ['2016', '2017', '2018', '2019', '2020']
    y_max = df.apply(lambda x: max(x[years_range]), axis=1)
    y_min = df.apply(lambda x: min(x[years_range]), axis=1)
    ax.fill_between(df.index, y_min, y_max, facecolor='grey', alpha=0.4)

    for year in ['2021', '2022']:  # '2017', '2018', '2019', '2020', '2021', '2022'
        plt.plot(df.index, df[year], label=year)

    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.title('Q2: Consumption Graph')
    plt.legend()
    plt.show()

    print('\nQ2:')
    print('Consumption PLOT is included with grey indicating the range across the 2016-2020 period')
    print('Comment: The trend for 2021 and 2022 energy consumption is for lower consumption than the previous years.')
    print('Comment: This trend may be caused by a combination of the COVID-19 pandemic and the war in Ukraine in 2022.')
    print('Comment: Consumption is much higher in the winnter months and lower in the summer months (for all years)')
    pass


def main_question_3():
    """
    -> The first word indicates direction and the number shows steps. The robot will stop moving with instruction “STOP”.
    -> Please write a function, which accepts instructions as a list.
    -> When first “STOP” instruction is given, it calculates the distance of Robot from the original position (0,0)
    """

    class Robot:
        def __init__(self):
            self.x = 0
            self.y = 0
            self.active = False

        def move(self, direction, distance):
            if self.active:
                match direction:
                    case 'LEFT':
                        self.x = self.x - 1 * distance
                    case 'RIGHT':
                        self.x = self.x + 1 * distance
                    case 'UP':
                        self.y = self.y + 1 * distance
                    case 'DOWN':
                        self.y = self.y - 1 * distance
            pass

        def accept_instruction(self, instruction):
            if instruction == 'BEGIN':
                self.active = True
            if instruction == 'STOP':
                self.active = False
            if instruction not in ['BEGIN', 'STOP']:
                direction, distance = instruction.split(' ')
                self.move(direction, float(distance))
            pass

    def calculate_robot_distance(instruction_list):
        """
        This function calcualtes the robots distance from origin using a simple Robot Class.
        """
        robot = Robot()
        for instruction in instruction_list:
            robot.accept_instruction(instruction)
        return np.sqrt(robot.x ** 2 + robot.y ** 2)

    # This example instruction set should end up back at the origin
    example_instructions = [
        "DOWN 200",
        "BEGIN",
        "LEFT 3",
        "UP 5",
        "RIGHT 4",
        "DOWN 7",
        "UP 2",
        "LEFT 1",
        "STOP"
        "UP 100"
    ]

    # Calculate the distance travelled following the input instructions
    dist = calculate_robot_distance(example_instructions)
    print('\nQ3:')
    print('Example instructions provided (printed below), please add/replace on line 214-223.')
    print(example_instructions)
    print('Calculated robot distance: ', dist)
    return dist


def main_question_4(plot=True):

    def normalise_venues(x: str):
        """ Simple function to group together Emission Venue A and Venue B"""
        return x.replace('Emission - Venue A', 'Emission').replace('Emission - Venue B', 'Emission')

    def fetch_product_data(products):
        data = pd.read_csv('data/Trades.csv')
        data['Product'] = data['Product'].apply(normalise_venues)
        products = list(map(lambda x: normalise_venues(x), products))
        return data[data['Product'].isin(products)], products  # return the altered list of products too

    def process_trade_data_to_ohlc(begin, end, products, freq='H'):
        """ Take data from Trades.csv and process so that it could be used in a candlestick plot with
            open high low close volume data."""

        # Parse "products" argument, it needs to be in list format
        if isinstance(products, str):
            products = [products]

        # Fetch the correct data and format the fields correctly
        data, products = fetch_product_data(products)  # get the corresponding data, products
        data['datetime'] = data['TradeDateTime'].apply(lambda x: datetime.datetime.strptime(x, "%d/%m/%Y %H:%M"))
        data = data.sort_values('datetime')
        data = data.set_index('datetime', drop=False)

        # Only include data between "begin" and "end", where necessary change begin and end to datetime object
        begin = datetime.datetime.strptime(begin, "%d/%m/%Y %H:%M") if isinstance(begin, str) else begin
        end = datetime.datetime.strptime(end, "%d/%m/%Y %H:%M") if isinstance(end, str) else end
        mask = (data['datetime'] > begin) & (data['datetime'] <= end)
        data = data.loc[mask]

        if freq != '1D':  # Where frequecy isn't 1 Day, only look at data between 0700 and 1700
            data = data.between_time('7:00', '17:00')

        product_ohlc = {}
        for product in products:  # Calculate the OHLC data for each product
            product_data = data[data['Product'] == product]
            high = product_data.resample(freq, on='datetime')['Price'].max()        # High
            low = product_data.resample(freq, on='datetime')['Price'].min()         # Low
            open = product_data.resample(freq, on='datetime')['Price'].first()      # Open
            close = product_data.resample(freq, on='datetime')['Price'].last()      # Close
            volume = product_data.resample(freq, on='datetime')['Quantity'].sum()   # Volume
            product_ohlc_data = pd.DataFrame(data={'high': high, 'low': low, 'open': open, 'close': close, 'Volumne': volume}, index=high.index)
            product_ohlc_data = product_ohlc_data[~product_ohlc_data['high'].isna()]
            product_ohlc[product] = product_ohlc_data

            # returns the OHLC data, but in the form of a dictionery -> easierfor multiple products
        return product_ohlc     # the key is the name of the contract, the value is the DataFrame with OHLC data

    def plot_candlestick(data, title='Candlestick'):
        fig = go.Figure(data=go.Ohlc(x=data.index,
                                     open=data['open'],
                                     high=data['high'],
                                     low=data['low'],
                                     close=data['close']))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(title=title, yaxis_title='Price')
        fig.show()
        pass

    # This is where we can implement the settings that we want to set for the generation of OHLC data
    frequency = 'H'  # "15MIN", "1D"
    products_of_interest = ['Emission - Venue A', 'Energy', 'Emission - Venue B']

    # Call the "process_trade_data_to_ohlc" function.
    ohlc = process_trade_data_to_ohlc(
        begin='18/04/2022 00:37',
        end='21/04/2022 22:07',
        products=products_of_interest,
        freq=frequency
    )

    # Optionally plot the different OHLC Candlestick plots
    if plot:
        for key, value in ohlc.items():
            plot_candlestick(value, title=key)

    print('\nQ4:')
    print('main_question_4() returns the OHLC data as a dictionery of "Product" : product_ohlc_DataFrame pairs.')
    print('An optional PLOT was included for each product in the given data to demonstrate the working code')
    return ohlc


if __name__ == '__main__':
    print('\nThis is Robert Hemming"s Interview Questions answered. This script runs with Python 3.10 and requires\n'
          'matplotlib and plotly. Furthermore, the "Consumption.csv", "Merge.csv" and "Trades.csv" should be stored in\n'
          'a directory named "data" at the same level as the python file assignment.py (which contains my solutions).')

    main_question_1()
    main_question_2()
    main_question_3()
    main_question_4()
