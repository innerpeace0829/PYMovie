import pandas as pd
import matplotlib.pyplot as plt

"""
從特定路徑path讀取檔案,並return一個df
"""
def load_data(path):
    df = pd.read_csv(path)
    df = df[df['global_step'] % 2000 != 0]
    df.insert(0, 'epoch', range(1, 1 + len(df)))
    return df

"""
根據df的結果,利用matplotlib繪圖
"""
def plot_data(df):
    plot = df.plot.line(x='epoch', y=['train_loss', 'eval_loss'])
    plt.title('Imdb rating')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'eval_loss'], loc='upper right')
    plt.ylim(0.0,5.0) 
    plt.show()

"""
若此段程式碼不是被import,則執行下面程式
"""
if __name__ == '__main__':
    path = 'your/path/to/training_progress_scores.csv'
    df = load_data(path)
    print(df[['epoch', 'f1_score']])
    plot_data(df)
