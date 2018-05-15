# DSAI HW3  : Adder & Subtractor Practice

# Environment
- windows 7
- python 3.5.4
- keras 2.0.8
- pytorch 0.4.0


# �e��
�b�g�o�ӧ@�~���ɭԧڥ�pytorch�g�F���model�M�Q��keras(sample code)�g�F�@��model
�̫�ϥΪ�model��keras������,�]���u��keras train�o�_��
- seq2seq pytorch (1)
- one layer RNN(GRU) pytorch (2)
- one layer LSTM keras  (3)
�������D������(2)�M(3)�ڨϥΪ��[�c�򥻤W�t���h,���O(2)�X�Gtrain���_�� training set ��loss�����U�h,�btraining set��accuracy�q�Ӥ��W�L10%�L,���Mdebug�F�ܤ[�٬O�����D�o�ͤF������D,�̫�~������sample code�ӧ�C
(1),(2)model��code�|��bgithub�W,���|��b���Prepositoty,URL�|��b�᭱�C
�ϥΪ�pytorch������0.4.0


# IDEA
# one layer LSTM keras
 �򥻤W�N�O��sample code���,�[�J-�o��token,�٦���training data size�ܨ⭿�C
 
# one layer RNN(GRU) pytorch
�Msample code�t���h LSTM cell���� GRU cell,�t�~����0123456...+-�o��character token embedding(randomly initialize jointly train with model parameters)
�o���I������P,���O�ڹ��չL��one hot train�٬Otrain���_��...

# seq2seq pytorch
�o��model��encoder�Mdecoder���OGRU,input/output sequence���g�Lmasking���B�z�קKpadding�v�T��V�m,encoder�R�X�Ӫ��Ĥ@��vector
����²�檺attention���o��,���骺���k�Ǭ���U��digit��embedding(randomly initialize jointly train with model parameters)�Moperation��embedding��attention�o��(�u�|���o����),���ĥΦb�C�@��time step decode���ɭԳ�������attention�O�]����ı�o�o�Ӱ��D��������ݭn�Ψ쨺���������k�C�o��model train�o����...

(2)(3)��data format�Msample code����
(1)�Msample code���P,���L�]��train���_�ӴN���S�O�����F�C

# ����
���զbValidaition set �W�����T�v
python main.py

# �s��
(1) : https://github.com/kumiko-oreyome/Seq2Seq-Adder-Subtractor
(2) : https://github.com/kumiko-oreyome/Adder-Subtractor_GRU
