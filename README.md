# Box_Office_Prediction
Main side project at iSpan BDSE (2022/11~2023/5).</br>
------------------------------------------------------
> 1. 透過爬蟲工具selenium抓取tmdb & imdb網站資料。
> 2. 透過python的numpy 和 pandas對資料進行清整與分析。
> 3. 透過特徵挑選挑選出有興趣的欄位。
> 4. 透過scikit-learn將其丟入模型訓練。 
> 5. 最優結果得出XGBoost R2 score 0.72。
> 6. 使用html css flask等技術將成果以網頁呈現。
本專題對票房預測尚算及格,然評分預測卻不盡理想,因評分項目容易因個人差異而有所變化,較難有客觀給分水準,故10分類的評分(加上NLP判斷空值label)較難準確,僅1與10分類有達到8成以上準確
