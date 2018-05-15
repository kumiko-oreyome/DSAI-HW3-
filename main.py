import numpy as np
class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)

def evaluation(model,Xs,ys):
    chars = '0123456789+- '
    ctable = CharacterTable(chars)
    right = 0
    preds = model.predict_classes(Xs, verbose=0)
    for i in range(len(preds)):
        correct = ctable.decode(ys[i])
        guess = ctable.decode(preds[i], calc_argmax=False)
        if correct == guess:
            right += 1
    print("MSG : Accuracy is {}".format(right / len(preds)))
    return right / len(preds)


if __name__ == '__main__':

    import argparse
    from keras.models import load_model
    import pickle as pkl

    parser = argparse.ArgumentParser()
    parser.add_argument('-mp','--model_path',
                       default='./model.h5',
                       help='model path')
    parser.add_argument('-cp','--corpus_path',
                       default='./corpus.pkl',
                       help='corpus path')
    args = parser.parse_args()
    print('loading model')
    model =  load_model(args.model_path)

    print('loading corpus')
    with open(args.corpus_path,'rb') as f:
        train_X,train_y,val_X,val_y = pkl.load(f)

    print('evaluation on validation set')
    print('accuracy = %.5f'%(evaluation(model,val_X,val_y)))



