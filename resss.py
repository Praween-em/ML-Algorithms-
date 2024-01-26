import pandas as pd


def Finds(data):
    hypothesis =["0"]*(len(data.columns)-1)
    
    for index, row in data.iterrows():
        for i in range(len(row)-1):
            if(row.iloc[-1]=="yes"):
                if hypothesis[i] =="0":
                    hypothesis[i]=row[i]
                elif hypothesis[i] != row[i]:
                    hypothesis[i]="@"
    
    return hypothesis

file_path = "enjoysport.csv"
data = pd.read_csv(file_path)
print(data)
print("hypothesis",Finds(data))