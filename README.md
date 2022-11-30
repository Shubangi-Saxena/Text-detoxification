# Text-detoxification
What is the problem our project aims to solve?
Social media has become an inherent part of our lives. People on these platforms choose to spread negativity by indulging in bullying, trolling online. Very harsh and hurtful words are written using offensive words and more. Fair, valuable and substantial points can be made without having to use toxic language. Hence, our aim is to try and rephrase these toxic offensive messages. 
Our aim is to paraphrase the sentence (toxic -> non-toxic) while preserving the meaning of it.

Txt Style Transfer algorithms can be developed in a supervised way with parallel corpora, i.e. text that comes twice with the same content but with different styles, and in an unsupervised way with non-parallel corpora. Parallel corpora are usually more difficult to get. Detoxification
needs better preservation of the original meaning than many other style transfer tasks, such as sentiment transfer, so it should be performed differently. There are many models that can be implemented such as, BART, ParaGedi, CondBERT and more.
We have implemented ParaGedi and CONDBert.
