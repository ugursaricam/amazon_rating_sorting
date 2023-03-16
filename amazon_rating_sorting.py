import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv('datasets/amazon_review.csv')
df = df_.copy()

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe().T

# Calculate the average score of the product.
df['overall'].mean() # 4.587589013224822

# Calculate the weighted average score by time.
df.info()

# instead of using 'day_diff'
df['reviewTime'] = pd.to_datetime(df['reviewTime'])

df['reviewTime'].max() # Timestamp('2014-12-07 00:00:00')

today_date = pd.to_datetime('2014-12-09 00:00:00') # max days +2 days

df['days'] = (today_date - df['reviewTime']).dt.days

df.sort_values('days', ascending=True)

df['days'].max() # 1065
df['days'].min() # 2

df['days'].quantile(0.25)
df['days'].quantile(0.5)
df['days'].quantile(0.75)

def time_based_weighted_average(dataframe,df_day, rating_column, w1=.28, w2=.26, w3=.24, w4=.22):
    last_30 = dataframe[dataframe[df_day] <= 30][rating_column].mean()
    between_30_180 = dataframe[(dataframe[df_day] > 30) & (dataframe[df_day] <= 180)][rating_column].mean()
    between_180_365 = dataframe[(dataframe[df_day] > 180) & (dataframe[df_day] <= 365)][rating_column].mean()
    older_than_365 = dataframe[dataframe[df_day] > 365][rating_column].mean()
    time_based_mean = last_30 * w1 + between_30_180 * w2 + between_180_365 * w3 + older_than_365 * w4
    return time_based_mean, last_30, between_30_180, between_180_365, older_than_365

time_based_mean, last_30, between_30_180, between_180_365, older_than_365 = time_based_weighted_average(df, 'days', 'overall')

# received more positive ratings and comments in recent days
time_based_mean     # 4.664898807388421
last_30             # 4.742424242424242
between_30_180      # 4.687378640776699
between_180_365     # 4.681397006414826
older_than_365      # 4.5216649607642445

df['overall'].mean() # 4.587589013224822

# Select 20 reviews for the product to be displayed on the product detail page.
df.sort_values('helpful_yes',ascending=False).head(20)

df['helpful_no'] = df['total_vote'] - df['helpful_yes']

df.sort_values('helpful_no',ascending=False)

# score_pos_neg_diff
def score_pos_neg_diff(up, down):
    return up - down

df['score_pos_neg_diff'] = df.apply(lambda x: score_pos_neg_diff(x['helpful_yes'],
                                                                 x['helpful_no']), axis=1)

# alt. df['score_pos_neg_diff'] = df['helpful_yes'] - df['helpful_no']

df.sort_values('score_pos_neg_diff',ascending=False).head(20)

# score_average_rating
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'],
                                                                     x['helpful_no']), axis=1)

# alt. df['score_average_rating'] = df['helpful_yes'] / (df['helpful_yes'] + df['helpful_no'])

df.sort_values('score_average_rating',ascending=False).head(20)

# wilson_lower_bound
def wilson_lower_bound(up, down, confidence=0.95):
    '''
    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    '''
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'],
                                                                 x['helpful_no']), axis=1)

df.sort_values('wilson_lower_bound',ascending=False).head(20)
