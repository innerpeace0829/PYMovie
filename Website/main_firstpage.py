from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
import numpy as np
import joblib
import columns_name as c_name
# 匯入模型
xgb_model = joblib.load('xgb_model.pkl')
scaler_for_model = joblib.load('scaler.pkl')


df = pd.read_csv('matchdata.csv')

# 建立首頁要的字典
list_genres = sorted(df[c_name.genres_pos0].unique())
list_company = sorted(df[c_name.company_pos0].unique())
list_director = sorted(df[c_name.director_pos0])
dict_genres = dict(zip(list_genres, list_genres))
dict_company = dict(zip(list_company, list_company))
dict_director = dict(zip(list_director, list_director))
keys_genres = dict_genres.keys()
keys_company = dict_company.keys()
keys_director = dict_director.keys()


# 建立route
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('main_firstpage.html', dict_genres=dict_genres, dict_company=dict_company, dict_director=dict_director, keys_genres=keys_genres, keys_company=keys_company, keys_director=keys_director)


@app.route('/get_genres0')
def get_genres0():
    global cond_genres0
    # 從請求中獲取選項值
    value = request.args.get('value')
    # 建立搜尋條件
    cond_genres0 = df[c_name.genres_pos0] == value
    # 將符合上述條件的 選單2 選項匯出由 變數list_genres2接收
    list_genres2 = df.loc[cond_genres0, c_name.genres_pos1]
    data = []
    # 資料處理成JSON格式: data = [{'value': 'aaa', 'text': 'bbb'}]
    for object1 in sorted(set(list_genres2)):
        dict_temp = dict()
        dict_temp['value'] = str(object1)
        dict_temp['text'] = str(object1)
        data.append(dict_temp)
    # 將數據轉換為 JSON 格式返回
    return jsonify(data)


@app.route('/get_genres1')
def get_data1():
    value = request.args.get('value')  # 從請求中獲取選項值
    cond_genres1 = df[c_name.genres_pos1] == value
    list_genres3 = df.loc[cond_genres0 & cond_genres1, c_name.genres_pos2]
    data = []
    for object1 in sorted(set(list_genres3)):
        dict_temp = dict()
        dict_temp['value'] = str(object1)
        dict_temp['text'] = str(object1)
        data.append(dict_temp)
    # 將數據轉換為 JSON 格式返回
    return jsonify(data)


@app.route('/get_company0')
def get_company0():
    global cond_company0
    value = request.args.get('value')  # 從請求中獲取選項值
    cond_company0 = df[c_name.company_pos0] == value
    list_company1 = df.loc[cond_company0, c_name.company_pos1]
    data = []
    for object1 in (sorted(set(list_company1))):
        dict_temp = dict()
        dict_temp['value'] = str(object1)
        dict_temp['text'] = str(object1)
        data.append(dict_temp)
    return jsonify(data)


@app.route('/result', methods=['GET'])
def result():

    # 取得前端變數value budget和sequal放後面
    genres0_value = request.args.get('genres0')
    genres1_value = request.args.get('genres1')
    genres2_value = request.args.get('genres2')
    company0_value = request.args.get('company0')
    company1_value = request.args.get('company1')
    director_value = request.args.get('director')

    # 設定篩選條件
    model_genres0_cond = df[c_name.genres_pos0] == genres0_value
    model_genres1_cond = df[c_name.genres_pos1] == genres1_value
    model_genres2_cond = df[c_name.genres_pos2] == genres2_value
    model_company0_cond = df[c_name.company_pos0] == company0_value
    model_company1_cond = df[c_name.company_pos1] == company1_value
    model_director_cond = df[c_name.director_pos0] == director_value

    # 模型要用的資料處理
    model_sequal_value = int(request.args.get('sequal'))
    model_budget_value = int(request.args.get('budget'))
    model_director_value = df.loc[model_director_cond,
                                  c_name.director_freq].iloc[0]
    model_genres_value = df.loc[model_genres0_cond & model_genres1_cond &
                                model_genres2_cond, c_name.genres_freq].iloc[0]
    model_company_value = df.loc[model_company0_cond &
                                 model_company1_cond, c_name.production_companies_freq].iloc[0]
    model_sequal_value = int(model_sequal_value)
    model_budget_value = int(model_budget_value)
    model_director_value = int(model_director_value)
    model_genres_value = int(model_genres_value)
    model_company_value = int(model_company_value)

    list_for_scaler = [model_budget_value, model_sequal_value,
                       model_genres_value, model_company_value, model_director_value]
    list_for_scaler = np.array(list_for_scaler).reshape(1, -1)
    print(list_for_scaler)
    X = scaler_for_model.transform(list_for_scaler)
    y = xgb_model.predict(X)
    revenue = int((int(np.exp(y)[0]))/10000)
    budget = int(model_budget_value/10000)
    margin = revenue - budget

    if budget == 10000:
        budget = '1億'
    else:
        budget = f'{str(budget)}萬'

    if revenue >= 10000:
        revenue = f'{str(revenue)[0]}億{str(revenue)[-4:]}萬'
    else:
        revenue = f'{str(revenue)}萬'

    if margin >= 10000:
        margin = f'{str(margin)[0]}億{str(margin)[-4:]}萬'
    else:
        margin = f'{str(margin)}萬'

    # 回傳資料但不刷新頁面
    return jsonify({'budget': budget, 'revenue': revenue, 'margin': margin})
    # 覆蓋 但使用者選的會沒紀錄
    # return render_template ('main_firstpage.html',revenue=revenue,budget=budget,margin=margin,dict_genres = dict_genres , dict_company = dict_company , dict_director = dict_director , keys_genres = keys_genres , keys_company = keys_company , keys_director = keys_director)


if __name__ == '__main__':
    app.run(debug=True)
