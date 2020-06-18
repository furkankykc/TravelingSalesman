import pandas as pd

data_name = 'data/mona-20k'
with open('{}.txt'.format(data_name), 'r') as data:
    sehir_data = data.readlines()
distlist = []
for i, s in enumerate(sehir_data):
    x, y = map(float, s.split())
    distance = []
    for j, t in enumerate(sehir_data):
        x_, y_ = map(float, t.split())
        dist = ((x - x_) ** 2 + (y - y_) ** 2) ** 0.5
        print('\rLoading %{:.2f}'.format((i * len(sehir_data) + j) * 100 / (len(sehir_data) * len(sehir_data))), end='')
        distance.append(dist)
    distlist.append(distance)
df = pd.DataFrame(data=distlist)
# df.to_csv('data/{}.csv'.format(data_name), sep=' ', header=False, float_format='%.2f', index=False)
