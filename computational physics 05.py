import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.callbacks import EarlyStopping

# 원본 데이터를 리스트 형태로 저장
data = [
    [2000, -1.7, -1.4, -1.1, -0.8, -0.7, -0.6, -0.6, -0.5, -0.5, -0.6, -0.7, -0.7],
    [2001, -0.7, -0.5, -0.4, -0.3, -0.3, -0.1, -0.1, -0.1, -0.2, -0.3, -0.3, -0.3],
    [2002, -0.1, 0.0, 0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.1],
    [2003, 0.9, 0.6, 0.4, 0.0, -0.3, -0.2, 0.1, 0.2, 0.3, 0.3, 0.4, 0.4],
    [2004, 0.4, 0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7],
    [2005, 0.6, 0.6, 0.4, 0.4, 0.3, 0.1, -0.1, -0.1, -0.1, -0.3, -0.6, -0.8],
    [2006, -0.9, -0.8, -0.6, -0.4, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 0.9, 0.9],
    [2007, 0.7, 0.2, -0.1, -0.3, -0.4, -0.5, -0.6, -0.8, -1.1, -1.3, -1.5, -1.6],
    [2008, -1.6, -1.5, -1.3, -1.0, -0.8, -0.6, -0.4, -0.2, -0.2, -0.4, -0.6, -0.7],
    [2009, -0.8, -0.8, -0.6, -0.3, 0.0, 0.3, 0.5, 0.6, 0.7, 1.0, 1.4, 1.6],
    [2010, 1.5, 1.2, 0.8, 0.4, -0.2, -0.7, -1.0, -1.3, -1.6, -1.6, -1.6, -1.6],
    [2011, -1.4, -1.2, -0.9, -0.7, -0.6, -0.4, -0.5, -0.6, -0.8, -1.0, -1.1, -1.0],
    [2012, -0.9, -0.7, -0.6, -0.5, -0.3, 0.0, 0.2, 0.4, 0.4, 0.3, 0.1, -0.2],
    [2013, -0.4, -0.4, -0.3, -0.3, -0.4, -0.4, -0.4, -0.3, -0.3, -0.2, -0.2, -0.3],
    [2014, -0.4, -0.5, -0.3, 0.0, 0.2, 0.2, 0.0, 0.1, 0.2, 0.5, 0.6, 0.7],
    [2015, 0.5, 0.5, 0.5, 0.7, 0.9, 1.2, 1.5, 1.9, 2.2, 2.4, 2.6, 2.6],
    [2016, 2.5, 2.1, 1.6, 0.9, 0.4, -0.1, -0.4, -0.5, -0.6, -0.7, -0.7, -0.6],
    [2017, -0.3, -0.2, 0.1, 0.2, 0.3, 0.3, 0.1, -0.1, -0.4, -0.7, -0.8, -1.0],
    [2018, -0.9, -0.9, -0.7, -0.5, -0.2, 0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 0.8],
    [2019, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.3, 0.1, 0.2, 0.3, 0.5, 0.5],
    [2020, 0.5, 0.5, 0.4, 0.2, -0.1, -0.3, -0.4, -0.6, -0.9, -1.2, -1.3, -1.2],
    [2021, -1.0, -0.9, -0.8, -0.7, -0.5, -0.4, -0.4, -0.5, -0.7, -0.8, -1.0, -1.0],
    [2022, -1.0, -0.9, -1.0, -1.1, -1.0, -0.9, -0.8, -0.9, -1.0, -1.0, -0.9, -0.8],
    [2023, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.1, 1.3, 1.6, 1.8, 1.9, 2.0],
    [2024, 1.8, 1.5, 1.1, 0.7]
]

# 데이터 변환
records = []
for row in data:
    year = row[0]
    for month, value in enumerate(row[1:], 1):
        records.append({'Year-Month': f'{year}-{month:02}', 'ONI': value})

# 데이터프레임 생성
df1 = pd.DataFrame(records)
df1.set_index('Year-Month', inplace=True)

# 주어진 데이터를 리스트 형태로 저장
data = [
    [2000, 1, -.50, -.96, -1.28],
    [2000, 2, -.26, -.69, -.91],
    [2000, 3, -.04, -.41, -.64],
    [2000, 4, .22, -.08, -.31],
    [2000, 5, .34, .04, -.18],
    [2000, 6, .34, .15, .08],
    [2000, 7, .25, .09, .03],
    [2000, 8, .14, .04, .00],
    [2000, 9, .08, .00, -.12],
    [2000, 10, -.11, -.30, -.37],
    [2000, 11, -.30, -.54, -.67],
    [2000, 12, -.36, -.72, -.96],
    [2001, 1, -.07, -.44, -.56],
    [2001, 2, -.04, -.41, -.63],
    [2001, 3, .22, -.03, -.29],
    [2001, 4, .42, .29, .26],
    [2001, 5, .27, .13, .11],
    [2001, 6, .33, .29, .46],
    [2001, 7, .43, .51, .61],
    [2001, 8, .09, .10, .12],
    [2001, 9, .22, .24, .35],
    [2001, 10, .17, .18, .28],
    [2001, 11, .18, .18, .22],
    [2001, 12, .22, .23, .17],
    [2002, 1, .49, .64, .95],
    [2002, 2, .51, .64, .78],
    [2002, 3, .45, .60, .55],
    [2002, 4, .23, .29, .32],
    [2002, 5, .10, .03, .07],
    [2002, 6, .43, .42, .67],
    [2002, 7, .53, .58, .73],
    [2002, 8, .70, .81, 1.05],
    [2002, 9, .93, 1.14, 1.41],
    [2002, 10, 1.04, 1.40, 1.72],
    [2002, 11, .78, 1.23, 1.58],
    [2002, 12, .26, .63, .74],
    [2003, 1, -.10, .13, .27],
    [2003, 2, -.34, -.27, -.11],
    [2003, 3, -.29, -.24, -.06],
    [2003, 4, -.44, -.51, -.49],
    [2003, 5, -.60, -.77, -.85],
    [2003, 6, .08, .03, .13],
    [2003, 7, .45, .44, .53],
    [2003, 8, .24, .20, .03],
    [2003, 9, .28, .16, .10],
    [2003, 10, .42, .33, .34],
    [2003, 11, .52, .53, .54],
    [2003, 12, .30, .25, .17],
    [2004, 1, .26, .18, .05],
    [2004, 2, .20, .16, .19],
    [2004, 3, .01, -.05, -.10],
    [2004, 4, .20, .19, .21],
    [2004, 5, .17, .16, .30],
    [2004, 6, .22, .14, .04],
    [2004, 7, .53, .62, .83],
    [2004, 8, .56, .61, .78],
    [2004, 9, .62, .70, .87],
    [2004, 10, .54, .56, .61],
    [2004, 11, .53, .58, .78],
    [2004, 12, .59, .64, .79],
    [2005, 1, .40, .42, .52],
    [2005, 2, .29, .42, .59],
    [2005, 3, .49, .72, 1.27],
    [2005, 4, .24, .36, .49],
    [2005, 5, .06, .01, .00],
    [2005, 6, .10, .04, .11],
    [2005, 7, .01, -.08, -.20],
    [2005, 8, -.05, -.17, -.42],
    [2005, 9, -.02, -.16, -.33],
    [2005, 10, .01, -.11, -.14],
    [2005, 11, -.21, -.44, -.57],
    [2005, 12, -.28, -.57, -.74],
    [2006, 1, -.28, -.67, -.97],
    [2006, 2, -.19, -.56, -.92],
    [2006, 3, .09, -.22, -.29],
    [2006, 4, .46, .25, .42],
    [2006, 5, .64, .49, .54],
    [2006, 6, .76, .71, .76],
    [2006, 7, .69, .74, .73],
    [2006, 8, .80, .91, 1.05],
    [2006, 9, .85, 1.01, 1.13],
    [2006, 10, .65, .77, .80],
    [2006, 11, .74, 1.00, 1.35],
    [2006, 12, .51, .68, .86],
    [2007, 1, -.18, -.20, -.46],
    [2007, 2, -.48, -.58, -.77],
    [2007, 3, -.47, -.62, -.72],
    [2007, 4, -.29, -.51, -.59],
    [2007, 5, -.23, -.49, -.58],
    [2007, 6, -.04, -.22, -.18],
    [2007, 7, -.10, -.32, -.48],
    [2007, 8, -.16, -.42, -.68],
    [2007, 9, -.35, -.69, -1.03],
    [2007, 10, -.52, -.87, -1.19],
    [2007, 11, -.54, -.97, -1.19],
    [2007, 12, -.49, -.87, -1.08],
    [2008, 1, -.50, -1.05, -1.50],
    [2008, 2, -.30, -.82, -1.20],
    [2008, 3, .21, -.26, -.45],
    [2008, 4, .48, .09, .02],
    [2008, 5, .60, .33, .17],
    [2008, 6, .65, .56, .38],
    [2008, 7, .50, .47, .42],
    [2008, 8, .17, .03, -.15],
    [2008, 9, -.12, -.40, -.69],
    [2008, 10, -.11, -.34, -.48],
    [2008, 11, -.37, -.65, -.77],
    [2008, 12, -.65, -1.08, -1.44],
    [2009, 1, -.32, -.77, -1.08],
    [2009, 2, .00, -.37, -.50],
    [2009, 3, .24, .00, .08],
    [2009, 4, .59, .52, .65],
    [2009, 5, .79, .77, .87],
    [2009, 6, 1.01, 1.07, 1.13],
    [2009, 7, .93, 1.04, 1.05],
    [2009, 8, .72, .79, .79],
    [2009, 9, .64, .72, .76],
    [2009, 10, .70, .86, 1.04],
    [2009, 11, 1.00, 1.31, 1.75],
    [2009, 12, .96, 1.28, 1.36],
    [2010, 1, .66, .94, 1.14],
    [2010, 2, .60, .93, 1.24],
    [2010, 3, .41, .65, .97],
    [2010, 4, -.06, -.01, -.06],
    [2010, 5, -.66, -.83, -1.00],
    [2010, 6, -.83, -1.12, -1.34],
    [2010, 7, -.80, -1.14, -1.36],
    [2010, 8, -.88, -1.33, -1.74],
    [2010, 9, -.92, -1.45, -1.93],
    [2010, 10, -.89, -1.47, -1.92],
    [2010, 11, -.76, -1.31, -1.64],
    [2010, 12, -.67, -1.24, -1.56],
    [2011, 1, -.35, -.92, -1.27],
    [2011, 2, .30, -.11, -.22],
    [2011, 3, .72, .50, .50],
    [2011, 4, .89, .72, .58],
    [2011, 5, .74, .64, .47],
    [2011, 6, .52, .46, .39],
    [2011, 7, .25, .20, .06],
    [2011, 8, -.07, -.22, -.54],
    [2011, 9, -.37, -.64, -1.01],
    [2011, 10, -.51, -.87, -1.26],
    [2011, 11, -.36, -.71, -.92],
    [2011, 12, -.40, -.81, -1.07],
    [2012, 1, -.28, -.79, -1.17],
    [2012, 2, .16, -.26, -.46],
    [2012, 3, .40, .06, .00],
    [2012, 4, .67, .51, .27],
    [2012, 5, .67, .58, .47],
    [2012, 6, .71, .66, .56],
    [2012, 7, .73, .76, .82],
    [2012, 8, .69, .74, .83],
    [2012, 9, .40, .44, .36],
    [2012, 10, .31, .38, .40],
    [2012, 11, .15, .22, .34],
    [2012, 12, -.11, -.17, -.27],
    [2013, 1, -.24, -.47, -.59],
    [2013, 2, -.05, -.23, -.17],
    [2013, 3, .13, -.04, .06],
    [2013, 4, .18, -.02, -.06],
    [2013, 5, .15, -.09, -.14],
    [2013, 6, .31, .16, .26],
    [2013, 7, .34, .28, .41],
    [2013, 8, .30, .25, .32],
    [2013, 9, .40, .29, .38],
    [2013, 10, .37, .26, .15],
    [2013, 11, .45, .48, .62],
    [2013, 12, .33, .26, .26],
    [2014, 1, .18, .00, -.33],
    [2014, 2, .43, .42, .39],
    [2014, 3, .93, 1.21, 1.60],
    [2014, 4, 1.00, 1.27, 1.41],
    [2014, 5, .74, 1.00, .95],
    [2014, 6, .31, .39, .27],
    [2014, 7, .05, .02, -.18],
    [2014, 8, .36, .37, .39],
    [2014, 9, .55, .58, .64],
    [2014, 10, .51, .51, .53],
    [2014, 11, .62, .68, .90],
    [2014, 12, .50, .48, .54],
    [2015, 1, .28, .22, .15],
    [2015, 2, .54, .65, .83],
    [2015, 3, .85, 1.17, 1.52],
    [2015, 4, 1.05, 1.42, 1.74],
    [2015, 5, 1.03, 1.42, 1.53],
    [2015, 6, .87, 1.27, 1.51],
    [2015, 7, .92, 1.36, 1.69],
    [2015, 8, .99, 1.43, 1.97],
    [2015, 9, 1.04, 1.48, 1.80],
    [2015, 10, 1.04, 1.51, 1.91],
    [2015, 11, .92, 1.41, 1.78],
    [2015, 12, .58, 1.04, 1.20],
    [2016, 1, .44, .88, 1.25],
    [2016, 2, -.04, .31, .56],
    [2016, 3, -.52, -.33, -.31],
    [2016, 4, -.92, -.85, -.88],
    [2016, 5, -1.03, -1.08, -1.15],
    [2016, 6, -.86, -.97, -1.05],
    [2016, 7, -.63, -.68, -.76],
    [2016, 8, -.50, -.56, -.71],
    [2016, 9, -.48, -.55, -.71],
    [2016, 10, -.59, -.75, -.92],
    [2016, 11, -.35, -.53, -.62],
    [2016, 12, -.06, -.18, -.24],
    [2017, 1, .18, .07, .01],
    [2017, 2, .36, .30, .15],
    [2017, 3, .43, .38, .22],
    [2017, 4, .34, .28, .06],
    [2017, 5, .37, .36, .30],
    [2017, 6, .21, .22, .22],
    [2017, 7, .13, .15, .16],
    [2017, 8, -.19, -.21, -.40],
    [2017, 9, -.45, -.57, -.79],
    [2017, 10, -.54, -.77, -.97],
    [2017, 11, -.41, -.65, -.84],
    [2017, 12, -.31, -.54, -.75],
    [2018, 1, .01, -.17, -.16],
    [2018, 2, .29, .09, -.11],
    [2018, 3, .46, .44, .51],
    [2018, 4, .58, .62, .80],
    [2018, 5, .72, .75, .88],
    [2018, 6, .77, .80, .86],
    [2018, 7, .74, .73, .81],
    [2018, 8, .75, .73, .81],
    [2018, 9, .88, .93, .98],
    [2018, 10, 1.08, 1.29, 1.47],
    [2018, 11, .97, 1.20, 1.25],
    [2018, 12, .71, .88, .92],
    [2019, 1, .53, .62, .59],
    [2019, 2, .59, .76, .94],
    [2019, 3, .70, .91, 1.19],
    [2019, 4, .21, .39, .41],
    [2019, 5, .01, .09, .07],
    [2019, 6, -.08, .07, .24],
    [2019, 7, -.11, .03, .13],
    [2019, 8, -.14, -.06, -.08],
    [2019, 9, .07, .09, .00],
    [2019, 10, .38, .49, .70],
    [2019, 11, .34, .34, .26],
    [2019, 12, .37, .45, .35],
    [2020, 1, .33, .48, .49],
    [2020, 2, .32, .51, .53],
    [2020, 3, .10, .19, .33],
    [2020, 4, -.20, -.24, -.30],
    [2020, 5, -.51, -.70, -.92],
    [2020, 6, -.35, -.54, -.62],
    [2020, 7, -.12, -.25, -.18],
    [2020, 8, -.29, -.55, -.80],
    [2020, 9, -.35, -.63, -.87],
    [2020, 10, -.49, -.85, -1.11],
    [2020, 11, -.44, -.80, -1.04],
    [2020, 12, -.30, -.71, -.94],
    [2021, 1, -.21, -.71, -1.02],
    [2021, 2, .06, -.44, -.82],
    [2021, 3, .54, .29, .27],
    [2021, 4, .67, .56, .60],
    [2021, 5, .58, .57, .65],
    [2021, 6, .33, .34, .31],
    [2021, 7, -.09, -.22, -.40],
    [2021, 8, -.30, -.51, -.83],
    [2021, 9, -.56, -.89, -1.28],
    [2021, 10, -.69, -1.09, -1.59],
    [2021, 11, -.51, -.88, -1.09],
    [2021, 12, -.37, -.81, -1.12],
    [2022, 1, .09, -.18, -.21],
    [2022, 2, .25, .05, .12],
    [2022, 3, .04, -.20, -.47],
    [2022, 4, .07, -.22, -.34],
    [2022, 5, .08, -.15, -.10],
    [2022, 6, .35, .18, .31],
    [2022, 7, -.03, -.32, -.46],
    [2022, 8, -.23, -.63, -.96],
    [2022, 9, -.19, -.65, -1.00],
    [2022, 10, -.19, -.66, -1.00],
    [2022, 11, -.03, -.50, -.75],
    [2022, 12, .19, -.23, -.31],
    [2023, 1, .40, -.03, -.24],
    [2023, 2, .54, .17, .09],
    [2023, 3, 1.07, .88, .84],
    [2023, 4, 1.29, 1.24, 1.19],
    [2023, 5, 1.24, 1.24, 1.11],
    [2023, 6, 1.30, 1.40, 1.40],
    [2023, 7, .95, 1.07, 1.02],
    [2023, 8, .88, 1.06, 1.09],
    [2023, 9, .78, .94, 1.03],
    [2023, 10, .79, .97, 1.13],
    [2023, 11, .82, 1.13, 1.45],
    [2023, 12, .48, .78, 1.06],
    [2024, 1, .12, .29, .31],
    [2024, 2, -.07, -.03, -.16],
    [2024, 3, -.42, -.51, -.54],
    [2024, 4, -.50, -.67, -.81],
]

# 데이터프레임 생성
df2 = pd.DataFrame(data, columns=['Year', 'Month', 'SST 130E-80W', 'SST 160E-80W', 'SST 180W-100W'])

# 'Year'와 'Month'를 하나의 인덱스로 결합
df2['Year-Month'] = df2['Year'].astype(str) + '-' + df2['Month'].astype(str).str.zfill(2)
df2.set_index('Year-Month', inplace=True)
df2.drop(columns=['Year', 'Month'], inplace=True)

# 데이터 병합
data = pd.concat([df1, df2], axis = 1)
data.reset_index(inplace=True)
data['Year-Month'] = pd.to_datetime(data['Year-Month'])

# 특징과 타겟 변수 설정
features = ['SST 130E-80W','SST 160E-80W', 'SST 180W-100W']
target = 'ONI'  # 타겟 변수


# 히스토그램 조사
data.hist(bins=50, figsize=(20, 15))
plt.show()

# 상관행렬 계산
corr_matrix = data.corr()

# 상관행렬 Heatmap 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# 데이터 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + [target]])

# 모델 1

# 시계열 데이터 준비 함수
def create_sequences(data, target, sequence_length=30):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[features].iloc[i:i + sequence_length].values)
        targets.append(data[target].iloc[i + sequence_length])
    return np.array(sequences), np.array(targets)

sequence_length = 30
X, y = create_sequences(pd.DataFrame(scaled_data, columns=features + [target]), target, sequence_length)

sequence_length = 30
X, y = create_sequences(pd.DataFrame(scaled_data, columns=features + [target]), target, sequence_length)


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, len(features))))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# 예측
y_pred = model.predict(X_test)

# 예측 결과 역정규화 (원래 값으로 복원)
# X_test의 마지막 타임 스텝과 예측 값을 결합하여 역정규화
X_test_last_step = X_test[:, -1, :]
y_pred_rescaled = scaler.inverse_transform(
    np.concatenate((X_test_last_step, y_pred), axis=1)
)[:, -1]
y_test_rescaled = scaler.inverse_transform(
    np.concatenate((X_test_last_step, y_test.reshape(-1, 1)), axis=1)
)[:, -1]

# 테스트 데이터의 실제 날짜 정보 추출
date_index = data['Year-Month'][sequence_length + train_size:]


# 결과 시각화
plt.plot(date_index, y_test, label='True ONI')
plt.plot(date_index, y_pred, label='Predicted ONI')
plt.legend()
plt.xlabel('Date')
plt.ylabel('ONI')
plt.title('ONI Prediction vs True Values')
plt.xticks(rotation=45)
plt.show()

# 모델 학습 강화 #

# 모델 2

# 시계열 데이터 준비 함수
def create_sequences(data, target, sequence_length=30):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[features].iloc[i:i + sequence_length].values)
        targets.append(data[target].iloc[i + sequence_length])
    return np.array(sequences), np.array(targets)

sequence_length = 30
X, y = create_sequences(pd.DataFrame(scaled_data, columns=features + [target]), target, sequence_length)

# 훈련 데이터와 테스트 데이터 분리 (시간 순서를 유지)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 모델 빌드 함수
def build_model(optimizer, dropout_rate=0.2, lstm_units=50):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(sequence_length, X.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# 하이퍼파라미터 설정
optimizer = RMSprop(learning_rate=0.001)
dropout_rates = [0.2, 0.3, 0.4, 0.5]
lstm_units_list = [50, 100, 150]

best_loss = float('inf')
best_model = None

# 하이퍼파라미터 튜닝
for dropout_rate in dropout_rates:
    for lstm_units in lstm_units_list:
        model = build_model(optimizer, dropout_rate, lstm_units)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stopping])
        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test Loss: {loss} (Optimizer: {optimizer.get_config()["name"]}, Dropout: {dropout_rate}, LSTM Units: {lstm_units})')
        if loss < best_loss:
            best_loss = loss
            best_model = model

print(f'Best Test Loss: {best_loss}')

# 예측
y_pred = best_model.predict(X_test)

# 예측 결과 역정규화
X_test_last_step = X_test[:, -1, :]
y_pred_rescaled = scaler.inverse_transform(
    np.concatenate((X_test_last_step, y_pred), axis=1)
)[:, -1]
y_test_rescaled = scaler.inverse_transform(
    np.concatenate((X_test_last_step, y_test.reshape(-1, 1)), axis=1)
)[:, -1]

# 테스트 데이터의 실제 날짜 정보 추출
date_index = data['Year-Month'][sequence_length + train_size:]

# y_test_rescaled와 y_pred_rescaled의 길이를 동일하게 맞추기
min_length = min(len(y_test_rescaled), len(y_pred_rescaled), len(date_index))

# 결과 시각화
plt.plot(date_index[:min_length], y_test_rescaled[:min_length], label='True ONI')
plt.plot(date_index[:min_length], y_pred_rescaled[:min_length], label='Predicted ONI')
plt.legend()
plt.xlabel('Date')
plt.ylabel('ONI')
plt.title('ONI Prediction vs True Values')
plt.xticks(rotation=45)
plt.show()

# 4개월 예측 수행 함수
def predict_next_month(model, last_sequence):
    last_sequence = last_sequence.reshape((1, sequence_length, last_sequence.shape[1]))
    prediction = model.predict(last_sequence)
    return prediction[0, 0]

# 4개월 예측 수행
predictions = []
last_sequence = scaled_data[-sequence_length:, :-1]  # 마지막 sequence_length만큼의 특성 데이터

for i in range(4):
    next_month_prediction = predict_next_month(best_model, last_sequence)
    predictions.append(next_month_prediction)
    next_entry = np.append(last_sequence[1:], np.array([[next_month_prediction] * last_sequence.shape[1]]), axis=0)
    last_sequence = next_entry

# 예측 결과 역정규화
predictions_rescaled = []
for pred in predictions:
    last_sequence_full = np.concatenate((last_sequence[-1, :], [pred])).reshape(1, -1)
    prediction_rescaled = scaler.inverse_transform(last_sequence_full)[0, -1]
    predictions_rescaled.append(prediction_rescaled)

# 결과 출력
for i, pred in enumerate(predictions_rescaled, start=5):
    print(f'2024년 {i}월의 예측 ONI: {pred}')
