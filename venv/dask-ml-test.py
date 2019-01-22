from dask_ml.linear_model import LogisticRegression
import dask.dataframe as dd

# Read the file
df = dd.read_csv("C:\\repos\\dataset\\connekt\\classificacao_2.csv")
ncol = len(df.columns) # Number of Columns
nrow = len(df) # Number of Rows
x = df[['v_1','v_2','v_3','v_4','v_5','v_6']]
y = df[['target']]
test = x
#converting dataframe to array
datanew = x.values
t = y.values
testnew = test.values
print(datanew)
model = LogisticRegression()
model.fit(datanew, t)
pred = model.predict(testnew)
print(pred.roll())