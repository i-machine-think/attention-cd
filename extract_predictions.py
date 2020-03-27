#!/usr/bin/env python
import sys
import os
import torch
import argparse
# sys.path.append(os.path.abspath('../src/word_language_model'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    '../src/word_language_model')))
import data
import numpy as np
import h5py
import pickle
import pandas
import time
import copy
from tqdm import tqdm
from torch.autograd import Variable
from model import SHARNN

parser = argparse.ArgumentParser(
    description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('model', type=str, default='model.pt',
                    help='Meta file stored once finished training the corpus')
parser.add_argument('-i', '--input', required=True,
                    help='Input sentences in Tal\'s format')
parser.add_argument('--data', default='data/wikitext/')
# parser.add_argument('-v', '--vocabulary', default='reduced_vocab.txt')
parser.add_argument('-o', '--output', default='output/', help='Destination for the output vectors')
parser.add_argument('--perplexity', action='store_true', default=False)
parser.add_argument('--eos-separator', default='</s>')
parser.add_argument('--fixed-length-arrays', action='store_true', default=False,
        help='Save the result to a single fixed-length array')
parser.add_argument('--format', default='npz', choices=['npz', 'hdf5', 'pkl'])
parser.add_argument('--unk-token', default='<unk>')
parser.add_argument('--use-unk', action='store_true', default=False)
parser.add_argument('--lang', default='en')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--uppercase-first-word', action='store_true', default=False)
args = parser.parse_args()

stime = time.time()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

def feed_input(model, hidden, mems, w):
    inp = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[w]]]))
    if args.cuda:
        inp = inp.cuda()
    # out, hidden = model(inp, hidden)
    output, hidden, mems = model(inp, hidden, mems=mems, return_h=False)
    output = model.decoder(output)
    return output, hidden, mems
def feed_sentence(model, h, mems, sentence):
    outs = []
    for w in sentence:
        out, h, mems = feed_input(model, h, mems, w)
        outs.append(torch.nn.functional.log_softmax(out[0], dim=-1).unsqueeze(0))
    return outs, h, mems

def model_load(fn, model):
    with open(fn, 'rb') as f:
        #torch.nn.Module.dump_patches = True
        #model, criterion, optimizer = torch.load(f)
        #model, criterion = torch.load(f)
        m, criterion = torch.load(f)
        d = m.state_dict()
        #del d['pos_emb']
        model.load_state_dict(d, strict=False)
        if False:
            for block in model.blocks:
                print(block.attn)
                if block.attn: block.attn.vq_collapse()
        del m
        return model, criterion

# Vocabulary
# vocab = data.Dictionary(args.vocabulary)
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

vocab = corpus.dictionary
ntokens = len(vocab)

# Sentences
sentences = [l.rstrip('\n').split(' ') for l in open(args.input + '.text', encoding='utf-8')]
gold = pandas.read_csv(args.input + '.gold', sep='\t', header=None, names=['verb_pos', 'correct', 'wrong', 'nattr'])

# Load model
print('Loading models...')
# import lstm
emsize = 650
nhid = 2600
nlayers = 4
dropout = 0.1
dropouth = 0.1
dropouti = 0.1
dropoute = 0.1
model = SHARNN('LSTM', ntokens, emsize, nhid, nlayers, dropout, dropouth, dropouti, dropoute, wdrop=0, tie_weights=True)
model, _ = model_load(args.model, model)
model.dropouti, model.dropouth, model.dropout, model.dropoute = dropouti, dropouth, dropout, dropoute
if args.cuda:
    model.cuda()

print('\nmodel: ' + args.model+'\n')
# model = torch.load(args.model, map_location=lambda storage, loc: storage)  # requires GPU model
# model.rnn.flatten_parameters()
# hack the forward function to send an extra argument containing the model parameters
# model.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)
# model_orig_state = copy.deepcopy(model.state_dict())

log_p_targets_correct = np.zeros((len(sentences), 1))
log_p_targets_wrong = np.zeros((len(sentences), 1))

stime = time.time()

output_fn = args.output + '.abl'


if args.lang == 'en':
    init_sentence = " ".join(["In service , the aircraft was operated by a crew of five and could accommodate either 30 paratroopers , 32 <unk> and 28 sitting casualties , or 50 fully equipped troops . <eos>",
                    "He even speculated that technical classes might some day be held \" for the better training of workmen in their several crafts and industries . <eos>",
                    "After the War of the Holy League in 1537 against the Ottoman Empire , a truce between Venice and the Ottomans was created in 1539 . <eos>",
                    "Moore says : \" Tony and I had a good <unk> and off-screen relationship , we are two very different people , but we did share a sense of humour \" . <eos>",
                    "<unk> is also the basis for online games sold through licensed lotteries . <eos>"])
elif args.lang == 'it':
    init_sentence = " ".join(['Ma altre caratteristiche hanno fatto in modo che si <unk> ugualmente nel contesto della musica indiana ( anche di quella \" classica \" ) . <eos>',
    'Il principio di simpatia non viene abbandonato da Adam Smith nella redazione della " <unk> delle nazioni " , al contrario questo <unk> allo scambio e al mercato : il <unk> produce pane non per far- ne dono ( benevolenza ) , ma per vender- lo ( perseguimento del proprio interesse ) . <eos>'])

            #init_sentence = " ".join(["Si adottarono quindi nuove tecniche basate sulla rotazione pluriennale e sulla sostituzione del <unk> con pascoli per il bestiame , anche per ottener- ne <unk> naturale . <eos>", "Una parte di questa agricoltura tradizionale prende oggi il nome di agricoltura biologica , che costituisce comunque una nicchia di mercato di una certa rilevanza e presenta prezzi <unk> . <eos>", "L' effetto estetico non scaturisce quindi da un mero impatto visivo : ad esempio , nelle architetture riconducibili al Movimento Moderno , lo spazio viene modellato sulla base di precise esigenze funzionali e quindi il raggiungimento di un risultato estetico deriva dal perfetto adempimento di una funzione . <eos>"])
else:
    raise NotImplementedError("No init sentences available for this language")

hidden = None
mems = None
init_out, init_h, init_mems = feed_sentence(model, hidden, mems, init_sentence.split(" "))

# Test: present prefix sentences and calculate probability of target verb.
for i, s in enumerate(tqdm(sentences)):
    out = None
    # reinit hidden
    #out = init_out[-1]
    hidden = init_h #model.init_hidden(1)
    mems = init_mems
    # intitialize with end of sentence
    # inp = Variable(torch.LongTensor([[vocab.word2idx['<eos>']]]))
    # if args.cuda:
    #     inp = inp.cuda()
    # out, hidden = model(inp, hidden)
    # out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)
    for j, w in enumerate(s):
        if j==0 and args.uppercase_first_word:
            w = w.capitalize()

        if w not in vocab.word2idx and args.use_unk:
            w = args.unk_token
        inp = Variable(torch.LongTensor([[vocab.word2idx[w]]]))
        if args.cuda:
            inp = inp.cuda()
        out, hidden, mems = model(inp, hidden, mems=mems, return_h=False)
        out = model.decoder(out)
        out = torch.nn.functional.log_softmax(out[0], dim=-1).unsqueeze(0)
        vp = gold.loc[i, 'verb_pos']
        vp += len(s) if vp < 0 else 0
        gold.loc[i,'verb_pos'] = vp
        if j==gold.loc[i,'verb_pos']-1:
            assert s[j+1] == gold.loc[i, 'correct'].lower()
            # Store surprisal of target word
            log_p_targets_correct[i] = out[0, 0, vocab.word2idx[gold.loc[i,'correct']]].item()
            log_p_targets_wrong[i] = out[0, 0, vocab.word2idx[gold.loc[i, 'wrong']]].item()
# Score the performance of the model w/o ablation
score_on_task = np.sum(log_p_targets_correct > log_p_targets_wrong)
p_difference = np.exp(log_p_targets_correct) - np.exp(log_p_targets_wrong)
score_on_task_p_difference = np.mean(p_difference)
score_on_task_p_difference_std = np.std(p_difference)

out = {
    'log_p_targets_correct': log_p_targets_correct,
    'log_p_targets_wrong': log_p_targets_wrong,
    'score_on_task': score_on_task,
    'accuracy_score_on_task': score_on_task,
    'sentences': sentences,
    'num_sentences': len(sentences),
    'nattr': list(gold.loc[:,'nattr']),
    'verb_pos': list(gold.loc[:, 'verb_pos'])
}

print(output_fn)
print('\naccuracy: ' + str(100*score_on_task/len(sentences)) + '%\n')
print('p_difference: %1.3f +- %1.3f' % (score_on_task_p_difference, score_on_task_p_difference_std))
# Save to file
if args.format == 'npz':
    np.savez(output_fn, **out)
elif args.format == 'hdf5':
    with h5py.File("{}.h5".format(output_fn), "w") as hf:
        for k,v in out.items():
            dset = hf.create_dataset(k, data=v)
elif args.format == 'pkl':
    with open(output_fn, 'wb') as fout:
        pickle.dump(out, fout, -1)

