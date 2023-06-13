# Multilingual TAC Relation Extraction Dataset

## Motivation
### For what purpose was the dataset created?
To enable more research on multilingual Relation Extraction, we generate translations of the TAC relation extraction
dataset using DeepL and Google Translate.

### Who created the dataset
The dataset was created by members of the
[DFKI SLT team: Leonhard Hennig, Philippe Thomas, Sebastian Möller, Gabriel Kressin](https://www.dfki.de/en/web/research/research-departments/speech-and-language-technology/speech-and-language-technology-staff-members)

## Composition

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?
The instances of this dataset are sentences from the
[original TACRED dataset](https://nlp.stanford.edu/projects/tacred/), which in turn
are sampled from the [corpus](https://catalog.ldc.upenn.edu/LDC2018T03) used in the yearly 
[TAC Knowledge Base Population (TAC KBP) challenges](https://tac.nist.gov/2017/KBP/index.html).  

### How many instances are there in total
In total, there are 1,422,914 instances in the dataset, on average 118,576 per language, 
including backtranslations of the test split.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?
Not applicable.

### What data does each instance consist of? 

- `id`: the instance id of this sentence, a `string` feature.
- `token`: the list of tokens of this sentence, a `list` of `string` features.
- `relation`: the relation label of this instance, a `string` classification label.
- `subj_start`: the 0-based index of the start token of the relation subject mention, an `ìnt` feature.
- `subj_end`: the 0-based index of the end token of the relation subject mention, exclusive, an `ìnt` feature.
- `subj_type`: the NER type of the subject mention, among the types used in the [Stanford NER system](https://stanfordnlp.github.io/CoreNLP/ner.html), a `string` feature.
- `obj_start`: the 0-based index of the start token of the relation object mention, an `ìnt` feature.
- `obj_end`: the 0-based index of the end token of the relation object mention, exclusive, an `ìnt` feature.
- `obj_type`: the NER type of the object mention, among 23 fine-grained types used in the [Stanford NER system](https://stanfordnlp.github.io/CoreNLP/ner.html), a `string` feature.

### Is any information missing from individual instances?
No.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?
Not applicable.

### Is there a label or target associated with each instance?
The target is the relation label.

### Are there recommended data splits (e.g., training, development/validation, testing)?
Yes, the train/dev/test splits of the translated versions correspond to the original TACRED data splits

### Are there any errors, sources of noise, or redundancies in the dataset?
Instances are drawn from a potentially noisy web / newswire corpus. Relation labels were assigned
using crowd workers, and have been shown to be partially erroneous, see e.g. 
[Alt et al., 2020](https://aclanthology.org/2020.acl-main.142/) and
[Stoica et al., 2021](https://arxiv.org/abs/2104.08398).

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?
The dataset is self-contained.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?
No.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?
The [authors](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf) of the original TACRED dataset
have not stated measures that prevent collecting sensitive or offensive text. Therefore, we do 
not rule out the possible risk of sensitive/offensive content in the translated data.

## Collection process
The [Github repository](https://github.com/DFKI-NLP/MultiTACRED) contains the code to generate
the dataset.

## Uses

### Has the dataset been used for any tasks already?
The dataset is used to train Relation Extraction models for evaluation of language-specific performance.

### Is there a repository that links to any or all papers or systems that use the dataset?
Please see [https://github.com/DFKI-NLP/MultiTACRED](https://github.com/DFKI-NLP/MultiTACRED)

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?
This dataset is generated from a dataset that is partially based on newswire data, licensed and distributed
by the Linguistic Data Consortium 
([https://catalog.ldc.upenn.edu/LDC2018T24](https://catalog.ldc.upenn.edu/LDC2018T24)). 
Therefore, this dataset will also
be distributed under an [LDC license](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf), 
and can only be used according to that license.

### Are there tasks for which the dataset should not be used?
No.

## Distribution
### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 
The dataset will be distributed via the Linguistic Data Consortium at [this URL](to-be-done)

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?
The dataset can be downloaded from [this URL](to-be-done) as set of JSON files.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?
This dataset is distributed under an [LDC license](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf).

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?
See the LDC license terms.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?
No.

## Maintenance

### Who is supporting/hosting/maintaining the dataset?
The dataset is hosted at LDC.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?
Please open an issue in the [Github repository](https://github.com/DFKI-NLP/MultiTACRED)

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?
We do not plan to update the dataset. Labeling errors can be corrected by applying the
patches made available by [TACRED Revisited](https://github.com/DFKI-NLP/tacrev) 
and/or [Re-TACRED](https://github.com/gstoica27/Re-TACRED/).