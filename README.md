# DSAI HW3  : Adder & Subtractor Practice

# Environment
- windows 7
- python 3.5.4
- keras 2.0.8
- pytorch 0.4.0


# 前言
在寫這個作業的時候我用pytorch寫了兩個model和利用keras(sample code)寫了一個model
最後使用的model為keras的版本,因為只有keras train得起來
- seq2seq pytorch (1)
- one layer RNN(GRU) pytorch (2)
- one layer LSTM keras  (3)
但不知道為什麼(2)和(3)我使用的架構基本上差不多,但是(2)幾乎train不起來 training set 的loss壓不下去,在training set的accuracy從來不超過10%過,雖然debug了很久還是不知道發生了什麼問題,最後才直接拿sample code來改。
(1),(2)model的code會放在github上,但會放在不同repositoty,URL會放在後面。
使用的pytorch版本為0.4.0


# IDEA
# one layer LSTM keras
 基本上就是把sample code改ㄧ改,加入-這個token,還有把training data size變兩倍。
 
# one layer RNN(GRU) pytorch
和sample code差不多 LSTM cell換成 GRU cell,另外有把0123456...+-這些character token embedding(randomly initialize jointly train with model parameters)
這兩點比較不同,但是我嘗試過用one hot train還是train不起來...

# seq2seq pytorch
這個model的encoder和decoder都是GRU,input/output sequence有經過masking的處理避免padding影響到訓練,encoder吐出來的第一個vector
有用簡單的attention的得到,具體的做法室為把各個digit的embedding(randomly initialize jointly train with model parameters)和operation的embedding做attention得到(只會做這ㄧ次),不採用在每一個time step decode的時候都做ㄧ次attention是因為不覺得這個問題有複雜到需要用到那麼複雜的方法。這個model train得更爛...

(2)(3)的data format和sample codeㄧ樣
(1)和sample code不同,不過因為train不起來就不特別說明了。

# 執行
測試在Validaition set 上的正確率
python main.py

# 連結
(1) : https://github.com/kumiko-oreyome/Seq2Seq-Adder-Subtractor
(2) : https://github.com/kumiko-oreyome/Adder-Subtractor_GRU
