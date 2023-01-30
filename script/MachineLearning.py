import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    input_fpath = sys.argv[1]
    result_fpath = sys.argv[2]

    df = pd.read_csv(input_fpath)
    df2 = df.drop(["マンション名", "都", "区", "区域",
                    "路線1","駅1","徒歩1",
                    "賃料","管理費","敷金","礼金"], 
                    axis=1)
    X = df2.drop("賃料+管理費", axis=1).values
    y = df["賃料+管理費"].values
    indices = np.array(range(X.shape[0]))

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices, test_size=0.3, random_state=0)    

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    predicts = pd.DataFrame(data=np.array([clf.predict(X_test)]).T, 
                            index=indices_test, 
                            columns=["予測価格"]) #予測価格
    diff = pd.DataFrame(data=np.array([y_test - clf.predict(X_test)]).T, 
                        index=indices_test, 
                        columns=["test-predict"]) #テスト-予測価格
    ratio = pd.DataFrame(data=np.array([np.round(np.abs((y_test - clf.predict(X_test)) / y_test)*100, 1)]).T, 
                         index=indices_test, 
                         columns=["ratio[%]"]) #価格差の割合
    df3 = pd.DataFrame(data=[df.iloc[index] for index in indices_test], 
                       index=indices_test)
    df4 = pd.concat([df3, predicts, diff, ratio], 
                    axis=1)
    #結果をcsvに書き込み
    df4.to_csv(result_fpath, index=False)
    print("[info] Wrote the result to {}".format(result_fpath))
    return 0

if __name__=="__main__":
    main()