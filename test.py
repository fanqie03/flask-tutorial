import pandas as pd

df = pd.DataFrame(data=pd.date_range('2019-01-01','2019-01-10'))

df.to_json()

print(df)
df.plot()
