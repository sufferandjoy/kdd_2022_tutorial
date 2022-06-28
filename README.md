# Deep Search Relevance Ranking - Session 3 
This repository is for Session 3 of KDD 2022 tutorial "Deep Search Relevance Ranking". In this session, we will walk through the major steps in knowledge distillation using:
* public dataset: we will experiment on the QADSM task in [XGLUE](https://huggingface.co/datasets/xglue) dataset, which is extracted from real Bing Ads traffic and available in [Hugging Face Datasets](https://huggingface.co/docs/datasets/index).
* public models: we implement a [BERT-Mini](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) model as our teacher model and a [TwinBERT](https://arxiv.org/abs/2002.06275) model built from two [BERT-Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) models as the corresponding student. Both teacher and student models could be easily implemented based on [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/index). 

## Knowledge Distillation
The most important paramter when calling main.py is **task**, which supports 4 different values in total:
* student_ft: this is the task to finetune a student model on human labels, where suffix **ft** means finetuning.
* teacher_ft: this is the task to finetune a teacher model on human labels.
* teacher_inf: this is the task to inference training data with the best teacher model which shows smallest average validaiton loss.
* student_kd: this is the task to distill teacher model's knowledge to the student model by regression to teacher score inference in teacher_inf task.

### Baseline: Directly Finetune Student using Human Labels
Before diving into knowledge distillation, it is necessary to get an idea on how well we are doing without knowledge distillaiton. For this reason, we firstly finetune our student model directly on human labels by setting task as **student_ft**. This would be used as our baseline later in the final evaluation.
```
python main.py --task student_ft --train_batch_size 512 --val_batch_size 2048
```
You may also use a different BERT encoder other than BERT-Tiny by setting **student_pretrained**, for example the following code snippet uses BERT-Mini:
```
python main.py --task student_ft --train_batch_size 512 --val_batch_size 2048 --student_pretrained google/bert_uncased_L-4_H-256_A-4
```

### Step 1: Finetune Teacher Model using Human Labels
Next, we will start from the very first step in knowledge distillation, which is to train a teacher model. Typical teacher models used in industrial applications are usually very powerful and hence resource-consuming, but we are often saved by the fact that we don't need to serve such models online. Instead, all the training and inference on these models would happen offline. Here, to train our teacher model we need to set task as **teacher_ft** and run the following code snippet, similar to what we do in the last step: 
```
python main.py --task teacher_ft --train_batch_size 512 --val_batch_size 2048
```
As with in the previous step, we may also switch to a different BERT encoder by setting **teacher_pretrained**.

### Step 2: Inference by Teacher Model
Once the above teacher_ft task completed, we can then infernece the entire training corpus to get teacher score on each training sample. The data set to be inferenced in this step is often refered to as distillation data, and it does not need to have human labels. That is why we can often leverage business logs in practical, industrial scenairos, since we usually have plenty of business logs and sampling from these logs is much easier and cheaper than labeling by human judges. 

Here we will do the inference on the same 100K training data used in the above finetuning steps. The scale of this data is much smaller than what we typically have in industrial scenarios (where we can sample billions of logs), but as we will see later, this facilitates a fair comparison between student_ft and student_kd:
```
python main.py --task teacher_inf --val_batch_size 4096
```
This operation would output a prediction.tsv file under output/teacher_inf. We need to copy this file to data/QADSM/, since this is where load_dataset.py would try to load the inferenced data in the next step.

### Step 3: Distill Knowledge from Teacher to Student
Finally it comes to the real distillation step! All we need to do is to run the code snippet once again with task **student_kd**:
```
python main.py --task student_kd --train_batch_size 512 --val_batch_size 2048
```

## Evaluation
now that all the 3 major steps for knowledge distillaiton have completed, we want to know whether all these efforts have end up with real impact. To see this, let us run our script for the last time with task **eval**, which will load the best checkpoint under teacher_ft, student_ft and student_kd settings respectively and conduct evaluation on the same test data:
```
python main.py --task eval
```
The above operation would print a table as below, where the last row highlights the improvement by comparing metrics for student_kd against that of student_ft. As we can see, even though we experiment on such a small data set with barely no advanced training strategies nor hyper-parameter tuning, we could see a 3% AUC lift:

| Task | PR AUC | ROC AUC |
| ---- | ---- | ---- |
| teacher_ft | 0.7166 | 0.7280 |
| student_ft | 0.6362 | 0.6504 |
| student_kd | 0.6585 | 0.6787 |
| delta      | 3.50% |  4.35% |
