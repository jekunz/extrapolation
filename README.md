## Instructions to reproduce results

Tested with Python 3.9.1. Needs packages: torch (tested with version 
1.8.0), transformers (4.3.3). Expects files 'en_ewt-ud-train.conllu' 
and 'en_ewt-ud-dev.conllu' in the same directory (unless the specs
are changed in create_objects_ud).

### Steps: 

1. Create files with Sentence objects by running create_objects_ud.

    Alternatively, replace pickle loader in main by:

    s_ld = read_conllu(i, 'en_ewt-ud-train.conllu'); 
    sentences = random.sample(s_ld, 1000); 
    s_ld_dev = read_conllu(i, 'en_ewt-ud-dev.conllu'); 
    sentences_dev = random.sample(s_ld_dev, 1000). 
    
    You will also need to import random; random.seed(42), 
    before entering the loop.

2. Specify the scoring function in main by modifying the last 
    import statement ("from scoring_functions.x import make_sets"). 
    The following options for x exist: 

    #### POS Tagging:
    
    * pos_sen_len_ling: Sentence length, linguistic splitting criterion
    * pos_sen_len_stat: Sentence length, distributional splitting criterion
    * pos_mft: Most frequent tag; binary
    * pos_tag_stats: Distribution of the tags
    * pos_loss: Loss-based ranking
    * pos_train_rank: Speed of learning
    
    #### Dependency Labelling:
    
    * stp_sen_len_ling: Sentence length, linguistic splitting criterion
    * stp_sen_len_stat: Sentence length, distributional splitting criterion
    * stp_arc_len: Arc length after with standard splitting point
    * stp_arc_len_mod: Arc length with the modified splitting 
    point (m_1 = 3 instead of m_1 = 2)
    * stp_loss: Loss-based ranking
    * stp_train_rank: Speed of learning
    
3. Specify the task in the 'task' variable in classifier: 'stp' for
    dependency labelling; 'pos' for POS tagging.
    
4.  Run main. If you are only interested in certain layers, 
    you can modify the loop accordingly. 
    
    
### More options: 

1. The control for the size of the data (section 3.3 of the paper) 
can be reproduced by running len_control. Modifications for 
on-the-fly data loading instead of using the pickle loader can be 
performed as described in point 1. 
2. Paths to other UD files can be specified in 
create_objects_ud. BERT models in other languages can be specified
in the BERT class init in word_representations. 