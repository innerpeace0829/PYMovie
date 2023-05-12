from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import time, pathlib, random
import torch
import os
import csv
from sklearn.metrics import f1_score, accuracy_score
import logging
from simpletransformers.classification import ClassificationModel
import sklearn
# def early_stopping_metric(eval_preds, eval_labels):
#     return f1_multiclass(eval_labels, eval_preds)
# def f1_multiclass(labels, preds): 
#     return f1_score(labels, preds, average='micro', zero_division=0)
def f1_multiclass(labels, preds): 
    return f1_score(labels, preds, average='weighted', zero_division=0)
def train():
    # 讀取資料後，開始訓練
    '''計時開始 - 訓練'''
    tStartTrain = time.time() 

    '''放置符合訓練格式的資料'''
    train_data = []

    '''讀取 AIdea 提供的訓練資料集'''
    train_aidea_csv = pd.read_csv('zzz.csv', escapechar='\\',header=None)


    '''資料轉換'''
    list_dataset_train_aidea = train_aidea_csv.values.tolist()
#     for dataset in list_dataset_train_aidea:
#         train_data.append([dataset[0], dataset[1]])
    train_data = [[dataset[0], dataset[1]] for dataset in list_dataset_train_aidea]
    
    # 切分資料
    '''隨機排序訓練資料'''
    random.shuffle(train_data) # 將訓練資料重新隨機排序
    middle = int(len(train_data) * 0.7) # 訓練資料的總數 (70%)
    list_train = train_data[:middle] # 透過 slicing 取得訓練資料 (70%)
    list_eval = train_data[middle:] # 透過 slicing 取得評估資料 (30%)





    # 模型手動設定
    '''pre-trained model、batch size 與 epoch'''
    model = 'roberta'
    model_name_prefix = ''
    model_name_main = 'roberta-base'
    model_name = model_name_prefix + model_name_main
    batch_size = 44
    epoch = 10

    '''output 資料夾'''
    output_dir = f"outputs/{model_name_main}-bs-{batch_size}-ep-{epoch}-cls-model/"

    '''自訂參數'''
    model_args = ClassificationArgs()
    model_args.train_batch_size = batch_size
    model_args.num_train_epochs = epoch
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.use_multiprocessing = True
    model_args.save_model_every_epoch = False # https://simpletransformers.ai/docs/tips-and-tricks/#dont-save-model-checkpoints
    model_args.save_steps = -1
    # model_args.learning_rate = 4e-5
    model_args.output_dir = output_dir
    # model_args.f1_score = True

    ''' Early Stopping 設定'''
    # https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
    # https://www.voxco.com/blog/matthewss-correlation-coefficient-definition-formula-and-advantages/
    model_args.use_early_stopping = False
    model_args.early_stopping_metric = "f1_score" # Matthews's correlation coefficient
    model_args.early_stopping_delta = 0.01 # test data does not improve upon the best mcc score by at least 0.01
    model_args.early_stopping_patience = 5 # 5 consecutive evaluations
    # model_args.early_stopping_metric = early_stopping_metric
    
    

    '''如果 use_early_stopping 為 False 的話，evaluate_during_training 可以改成 True'''
    model_args.evaluate_during_training = True
    model_args.eval_batch_size = batch_size
    model_args.use_multiprocessing_for_evaluation = True

    '''每幾個 steps 執行一次 eval'''
    model_args.evaluate_during_training_steps = 2000 # An evaluation will occur once for every 2000 training steps.

    '''建立 ClassificationModel'''
    model = ClassificationModel(
        model, 
        model_name, 
        use_cuda=torch.cuda.is_available(), #torch.cuda.is_available()
        cuda_device=0, # 使用第 1 顆 GPU (Device ID: 0),
        num_labels=10,
        args=model_args
    )

    '''train_data 轉成 data frame 後，給序欄位名稱'''
    train_df = pd.DataFrame(list_train)
    train_df.columns = ["text", "labels"]

    '''eval_data 轉成 data frame 後，給序欄位名稱'''
    eval_df = pd.DataFrame(list_eval)
    eval_df.columns = ["text", "labels"]

    '''訓練model'''
    model.train_model(train_df, eval_df=eval_df, f1_score=f1_multiclass)

    '''計時結束 - 訓練'''
    tEndTrain = time.time()

    '''訓練執行的時間'''
    print("[Train] It took %f sec" % (tEndTrain - tStartTrain))





    # 評估
    '''計時開始 - 評估'''
    tStartEval = time.time() 

    '''評估模型'''
    result, model_outputs, wrong_predictions = model.eval_model(eval_df,  f1_score=f1_multiclass) 
    print(f"result: {result}, model_outputs: {model_outputs}, wrong_predictions: {wrong_predictions}")

    '''計時結束 - 評估'''
    tEndEval = time.time()

    '''評估執行的時間'''
    print("[Evaluation] It took %f sec" % (tEndEval - tStartEval))





    # 預測
    '''計時開始 - 預測'''
    tStartPredict = time.time() 

    '''預測結果'''
    predictions, raw_outputs = model.predict([
        "I watched this film because I'm a big fan of River Phoenix and Joaquin Phoenix. I thought I would give their sister a try, Rain Phoenix. I regret checking it out. She was embarrasing and the film just has this weird plot if thats what you want to call it. Sissy was just weird and Jellybean just sits on a toilet who both sleep with this old man in the mountains, whats going on? I have never been so unsatisfied in my life. It was just total rubbish. I can't believe that the actors agreed to do such a waste of film, money, time and space. Have Sissy being 'beautiful' didnt get to me. I thought she was everything but that. Those thumbs were just stupid, and why do we care if she can hitchhike? WHATS THE POINT??? 0 out of 10, shame the poll doesnt have a 0, doesnt even deserve a 1. Hopefully, Rain is better in other films, I forgive her for this one performance, I mean I wouldnt do much better with that film.",
        "It does not seem that this movie managed to please a lot of people. First off, not many seem to have seen it in the first place (I just bumped into it by accident), and then judging by the reviews and the rating, of those that did many did not enjoy it very much.<br /><br />Well, I did. I usually tolerate Gere for his looks and his charm, and even though I did not consider him a great actor, I know he can do crazy pretty well (I liked his Mr Jones). But this performance is all different. He is not pretty in this one, and he is not charming. His character is completely different from anything I had seen from him up to that point---old, ugly, broken, determined. And Gere, in what to me is so far his best performance ever, pulls it off beautifully. I guess it is a sign of how well an actor does his job if you cannot imagine anyone else doing it instead---think Hopkins as Hannibal Lecter, or Washington as Alonzo in Training Day. That is how good Gere was here.<br /><br />The rest of the cast were fine by me, too. I guess I would not have cast Danes in this role, mostly because I think she is too good-looking for it. But she actually does an excellent job, holding her own with a Gere in top form, which is no small feat. Strickland easily delivers the best supporting act, in a part that requires a considerable range from her. I actually think she owns the key scene with Gere and Danes, and that is quite an achievement.<br /><br />So what about the rest of the movie, apart from some excellent acting? The story is perhaps not hugely surprising, some 8mm-ish aspects to it, but adding the ""veteran breaks in rookie"" storyline to the who-dunnit, and also (like Silence of the Lambs) adding a sense of urgency through trying to save the girl and the impending retirement of Gere's character. All that is a backdrop to the development of the two main characters, as they help each other settle into their respective new stations in life. That's a lot to accomplish in a 100 minutes, but it is done well, and we end up caring for the characters and what happens to them.<br /><br />Direction and photography were adequate. I could have done without the modern music-video camera movements and cutting, but then I am an old curmudgeon, and it really wasn't all that bad, in fact I think it did help with the atmosphere of the movie, which as you might have guessed, by and large isn't a happy one.<br /><br />Worth seeing.",
        "      Enough is not a bad movie , just mediocre .",
        "my friend and i rented this one a few nights ago. and, i must say, this is the single best movie i have ever seen. i mean, woah! ""dude, we better get some brew before this joint closes"" and ""dude, linda's not wearin' a bra again."" what poetry! woah! and it's such a wonderfuly original movie, too. i mean, you don't usually find a slasher film where every single murder is exactly the same. i mean, exactly! now that's originality. and almost all the transitions between scenes are these great close-ups of the psycho in the ER scrubs. how cool! the acting is so wonderful to. the dad was just brilliant. must have studied REAL DADS before filming. and how many movies do you find that just don't make any sense? not many. but this is one of those gems. i mean, how cool is it that one guy waited outside for like six hours to pull a prank, while his friends were both inside? that's really cool. overall i'd say this is the single greatest film of the genre, nay, in the world! *****",
        "Just about everything in this movie is wrong, wrong, wrong. Take Mike Myers, for example. He's reached the point where you realize that his shtick hasn't changed since his SNL days, over ten years ago. He's doing the same cutesy stream-of-consciousness jokes and the same voices. His Cat is painfully unfunny. He tries way to hard. He's some weird Type A comedian, not the cool cat he's supposed to be. The rest of the movie is just as bad. The sets are unbelievably ugly --- and clearly a waste of millions of dollars. (Cardboard cut-outs for the background buildings would have made more sense than constructing an entire neighborhood and main street.) Alec Balwin tries to do a funny Great Santini impression, but he ends up looking and sounding incoherent. There's even an innapropriate cheesecake moment with faux celebrity Paris Hilton --- that sticks in the mind simply because this is supposed to be a Dr. Seuss story. Avoid this movie at all costs, folks. It's not even an interesting train wreck. (I hope they'll make Horton Hears a Who with Robin Williams. Then we'll have the bad-Seuss movie-starring-spasitc- comedian trilogy.)",
        "this isn't 'Bonnie and Clyde' or 'Thelma and Louise' but it is a fine road movie. it sets up its two main characters gently and easily. viewers learn the underlying tensions quickly, which is a tribute to the director. there is the young french (and English) speaking son who wants to do well in France, has a french girlfriend and who drinks alcohol, parties as young men do. And there is his moroccan arabic (and french) speaking father who devoutly follows his Muslim faith, with generosity and the wisdom of elders and who rejects the new culture surrounding him (like mobile phones). the film could explore very powerful politics - the odd couple drive thru the former Yugoslavia, thru Turkey and then thru the Middle East to get to Mecca. these are areas where the Muslim populations have been involved in wars, repression, ethnic cleansing; where dictators have pursued torture and summary executions to hold power and where religious communities are in constant deadly battle with each other. yet the film moves thru those places and possibilities with only hints of such agendas. the relationship between the two is key to this film, and faith, politics are the backdrop. it seems to be saying that we are all human, and need to understand and care for each other in order to manage well in this world. it certainly isn't 'Natural Born Killers' and is all the better for it.",
        "I have to say that I really liked UNDER SIEGE and HARD TO KILL.<br /><br />watching Seagal doing his funny martial arts on people. I have<br /><br />been always looking forward to Seagal-Movies and, unfortunately, I was first disappointed by GLIMMER MAN, which I found really bad. THE FOREIGNER is probably one of the worst Seagal has ever<br /><br />acted in. Horribly boring, badly edited, wrongest soundtrack and so on! Dear reader, do yourself a favor an stay away from this!<br /><br />Honestly: Stay away!",
        "Kramer Vs. Kramer is a near-heartening drama about shocking, drastic augmentations of the two subjects of a failed married couple. Meryl Streep, in the throes of her trademark maternal sensitivity, plays an unhappy stay-at-home mother who feels confined to such a role and within the first five minutes of the film leaves her inattentive husband, in a fantastic performance by Dustin Hoffman, to find another role for herself. Hoffman is dumbstruck, having absolutely no idea what to do with himself, having taken so much for granted that he doesn't know the first thing about getting his son to school in the morning.<br /><br />Hoffman seamlessly characterizes this husband as such a juicy load of setbacks. He is restless, relentless and impatient, but even though the positive side to those three adjectives should include just the opposite, he is unremittingly fixated on whatever he turns his head to. He's been focused on his career in advertising, and when he is left to raise his son Billy all by himself, chaos ushers in immediately. He's the one throwing temper tantrums and quitting angrily halfway through an activity. After awhile, as he befriends his neighbor and Joanna's former friend, played by sexy Jane Alexander, Hoffman cools his jets enough to understand why his wife left. In the meantime, his boundless energy redirects towards raising Billy and he loses his job.<br /><br />The custody battle of the title is a brilliantly grey circumstance. Even if the ending is a little unmotivated, subjectified for the audience, the last line and the last shot still have that witty screen writing touch that seemed to diminish after the magical 1970s.",
        "Like the other comments says, this might be surprise to those who haven't seen the work of Jeunet & Caro or Emir Kusturica. But have you already seen Delicatessen, there is nothing new it this film. I thought Delicatessen was great when it came out, but this film just arrive too late to be of any interest. I don't think it's a worse film than Delicatessen but it's a bore to see it now, like it probably would be to watch Delicatessen again. There is really no point to the film, nothing that really matter or stays with you. There may be a distant similarity to the films of Kusturica, but he's really in a different league, so you should rather go see his films than waste your time on Tuvalu.",
        "The tunes are the best aspect of this television film which has admittedly better-than-average production values, but very surface and slightly altered biography. Dramatizes Richard's discovery of ""We've Only Just Begun"" and Karen's marriage troubles admirably (the ""Superstar"" montage was a nice touch), yet notably leaves out the disagreement with Neil Sedaka, the contribution of Tony Peluso's guitar solos, etc. Gibb is sweet in her Karen persona, but it doesn't include the tomboyish and gutsier sides of the real Carpenter's personality. Anderson is in fine form as the creative and take-charge Richard, and Fletcher makes her mark as the loving but overbearing Agnes. The most haunting moment of the original broadcast is the use of ""Goodbye to Love"" in the background of a commercial displaying an anorexia hotline."
    ])

    print(f"預測結果: {predictions}")
    # 答案: label => [0,1,0,0,0, 1,0,1,0,1]

    '''計時結束 - 預測'''
    tEndPredict = time.time()

    '''預測執行的時間'''
    print("[Predict] It took %f sec" % (tEndPredict - tStartPredict))


if __name__ == "__main__":
    train()