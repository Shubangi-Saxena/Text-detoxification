# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %%
import torch
import numpy as np

# %%
from importlib import reload
import gedi_adapter
reload(gedi_adapter)
from gedi_adapter import GediAdapter

# %%
import modeling_utils
import text_processing
import modeling_gpt2
import gedi_training

# %%
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
t5name = 'SkolkovoInstitute/t5-paraphrase-paws-msrp-opinosis-paranmt'

# %%
import sys
sys.path.append(os.path.abspath('../transfer_utils/'))

import text_processing
reload(text_processing);

# %%
tokenizer = AutoTokenizer.from_pretrained(t5name)

# %%
para = AutoModelForSeq2SeqLM.from_pretrained(t5name)
para.resize_token_embeddings(len(tokenizer)) 

# %%
model_path = 'SkolkovoInstitute/gpt2-base-gedi-detoxification'
gedi_dis = AutoModelForCausalLM.from_pretrained(model_path)

# %%
NEW_POS = tokenizer.encode('normal', add_special_tokens=False)[0]
NEW_NEG = tokenizer.encode('toxic', add_special_tokens=False)[0]

# %%
# add gedi-specific parameters
if os.path.exists(model_path):
    w = torch.load(model_path + '/pytorch_model.bin', map_location='cpu')
    gedi_dis.bias = w['bias']
    gedi_dis.logit_scale = w['logit_scale']
    del w
else:
    gedi_dis.bias = torch.tensor([[ 0.08441592, -0.08441573]])
    gedi_dis.logit_scale = torch.tensor([[1.2701858]])
print(gedi_dis.bias, gedi_dis.logit_scale)

from gedi_adapter import GediAdapter
import transformers
import modeling_utils
dadapter = GediAdapter(model=para, gedi_model=gedi_dis, tokenizer=tokenizer, gedi_logit_coef=5, target=1, neg_code=NEW_NEG, pos_code=NEW_POS, lb=None, ub=None)
text = 'The internal policy of Trump is flawed.'
print('====BEFORE====')
print(text)
print('====AFTER=====')
#input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(para.device)
inputs = tokenizer.encode(text, return_tensors='pt').to(para.device)
print("hey")
result=dadapter.generate(inputs, num_return_sequences=3, do_sample=True, temperature=1.0, repetition_penalty=3.0,num_beams=1)
for r in result:
    print(tokenizer.decode(r, skip_special_tokens=True))

# %%
import torch
#device = torch.device('cuda:0')
device = torch.device('cpu')

# %%
import gc

def cleanup():
    gc.collect()
    if torch.cuda.is_available() and device.type != 'cpu':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

# %%
para.to(device);
para.eval();


# %%
gedi_dis.to(device);
gedi_dis.bias = gedi_dis.bias.to(device)
gedi_dis.logit_scale = gedi_dis.logit_scale.to(device)
gedi_dis.eval();

# %%
with open('../../data/test/test_10k_toxic', 'r') as f:
    test_data = [line.strip() for line in f.readlines()]
print(len(test_data))

# %%
dadapter = GediAdapter(
    model=para, gedi_model=gedi_dis, tokenizer=tokenizer, gedi_logit_coef=10, target=0, neg_code=NEW_NEG, pos_code=NEW_POS, 
    reg_alpha=3e-5, ub=0.01
)

# %%
def paraphrase(text, n=None, max_length='auto', beams=2):
    texts = [text] if isinstance(text, str) else text
    texts = [text_processing.text_preprocess(t) for t in texts]
    inputs = tokenizer(texts, return_tensors='pt', padding=True)['input_ids'].to(dadapter.device)
    if max_length == 'auto':
        max_length = min(int(inputs.shape[1] * 1.1) + 4, 64)
    result = dadapter.generate(
        inputs, 
        num_return_sequences=n or 1, 
        do_sample=False, temperature=0.0, repetition_penalty=3.0, max_length=max_length,
        bad_words_ids=[[2]],  # unk
        num_beams=beams,
    )
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in result]
    texts = [text_processing.text_postprocess(t) for t in texts]
    if not n and isinstance(text, str):
        return texts[0]
    return texts

# %%
paraphrase(test_data[:3])

# %% [markdown]
# # Evaluate

# %%
from transformers import RobertaForSequenceClassification, RobertaTokenizer
clf_name = 'SkolkovoInstitute/roberta_toxicity_classifier_v1'
clf = RobertaForSequenceClassification.from_pretrained(clf_name).to(device);
clf_tokenizer = RobertaTokenizer.from_pretrained(clf_name)

# %%
import numpy

# %%
def predict_toxicity(texts):
    with torch.inference_mode():
        inputs = clf_tokenizer(texts, return_tensors='pt', padding=True).to(clf.device)
        out = torch.softmax(clf(**inputs).logits, -1)[:, 1].cpu().numpy()
    return out

# %%
predict_toxicity(['hello world', 'hello aussie', 'hello fucking bitch'])

# %% [markdown]
# # The baseline

# %%
# reload(gedi_adapter)
from gedi_adapter import GediAdapter


adapter2 = GediAdapter(
    model=para, gedi_model=gedi_dis, tokenizer=tokenizer, 
    gedi_logit_coef=10, 
    target=0, pos_code=NEW_POS, 
    neg_code=NEW_NEG,
    reg_alpha=3e-5,
    ub=0.01,
    untouchable_tokens=[0, 1],
)


def paraphrase(text, max_length='auto', beams=5, rerank=True):
    texts = [text] if isinstance(text, str) else text
    texts = [text_processing.text_preprocess(t) for t in texts]
    inputs = tokenizer(texts, return_tensors='pt', padding=True)['input_ids'].to(adapter2.device)
    if max_length == 'auto':
        max_length = min(int(inputs.shape[1] * 1.1) + 4, 64)
    attempts = beams
    out = adapter2.generate(
        inputs, 
        num_beams=beams,
        num_return_sequences=attempts, 
        do_sample=False, 
        temperature=1.0, 
        repetition_penalty=3.0, 
        max_length=max_length,
        bad_words_ids=[[2]],  # unk
        output_scores=True, 
        return_dict_in_generate=True,
    )
    results = [tokenizer.decode(r, skip_special_tokens=True) for r in out.sequences]

    if rerank:
        scores = predict_toxicity(results)
    
    results = [text_processing.text_postprocess(t) for t in results]
    out_texts = []
    for i in range(len(texts)):
        if rerank:
            idx = scores[(i*attempts):((i+1)*attempts)].argmin()
        else:
            idx = 0
        out_texts.append(results[i*attempts+idx])
    return out_texts

torch.manual_seed(0)
paraphrase(['fuck you!', 'you are stupid!', 'you remind me of the chump .', 'he has to be a terrorist ! .'], beams=3)

# %%
paraphrase(['fuck you!', 'you are stupid!', 'you remind me of the chump .', 'he has to be a terrorist ! .'], beams=3, rerank=False)

# %%
paraphrase(['fuck you!', 'you are stupid!', 'you remind me of the chump .', 'he has to be a terrorist ! .'], beams=10, rerank=False)

# %%
paraphrase(['fuck you!', 'you are stupid!', 'you remind me of the chump .', 'he has to be a terrorist ! .'], beams=10, rerank=True)

# %%
batch_size = 2

# %%
import os
from tqdm.auto import tqdm, trange

cleanup()

lines = test_data[:10]


for i in trange(int(len(lines) / batch_size + 1)):
    if i % 10 == 0:
        cleanup()
    t = i * batch_size
    batch = [line.strip() for line in lines[t:(t+batch_size)]]
    if not batch:
        continue

    try:
        res = paraphrase(batch, max_length='auto', beams=10)
    except RuntimeError:
        print('runtime error for batch ', i)
        try:
            cleanup()
            res = [paraphrase([text], max_length='auto', beams=3)[0] for text in batch]
        except RuntimeError:
            print('runtime error for batch ', i, 'even with batch size 1')
            res = batch
            cleanup()
    for out in res:
        print(out)

# %% [markdown]
# Expected output:
# 
# ```
# You'd be right. You'd be right. You'd
# As snooty and overbearing as its
# A bad society does the wrong things, and votes for the wrong politicians.
# President, he's the man. He's the man.
# Colberg's a bad guy, and I'm a TSA.
# Dubious, dubious, dubious, dubious.
# I like you. I think you'll be an intelligent man and your contributions will be a big deal.
# Luna's oh-so-stunnel, immature girl......who has no idea how to do it, which smells bad, I'd like to see
# Mormonis is the wrong thing to do. The wrong thing to do. The wrong thing to do. The wrong thing to do. The wrong thing to do. The right thing to do. The right thing to do. The right thing to do 
# You'd be a bad guy, uninitiated.
# ```

# %%



