# This is the entry script for KDD tutorial

import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description="Params")

    # overall config
    parser.add_argument("--model", type=str, default='teacher', help="teacher or student")
    parser.add_argument("--mode", type=str, default='train', help="train or inference")
    parser.add_argument("--use_labeled_data", type=str, default='true', help="Use labeled data or not.")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Batch size (per device) for the training.")
    parser.add_argument("--val_batch_size", type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--logfreq", type=int, default=50, help="Frequency to print training logs.")
    parser.add_argument("--output_dir", type=str, default='output', help="Where to store the final model.")
    # parser.add_argument("--prev_ckpt", type=str, default=r'output\teacher\model_best.pth', help="Checkpoint name.")
    parser.add_argument("--prev_ckpt", type=str, default=None, help="Checkpoint name.")

    # config on teacher model
    parser.add_argument("--teacher_pretrained", type=str, default='google/bert_uncased_L-4_H-256_A-4', help="Path to pretrained model for teacher.")
    parser.add_argument("--max_length", type=int, default=32, help="Maximum sequence length after tokenization.")

    # config on student model
    parser.add_argument("--student_pretrained", type=str, default='google/bert_uncased_L-2_H-128_A-2', help="Path to pretrained model for TwinBERT encoder")
    parser.add_argument("--max_length_query", type=int, default=9, help="Maximum query sequence length after tokenization.")
    parser.add_argument("--max_length_ad", type=int, default=24, help="Maximum ad sequence length after tokenization.")

    args = parser.parse_args()
    return args


class TeacherModel(nn.Module):
    def __init__(self, args):
        super(TeacherModel, self).__init__()
        self.model = BertModel.from_pretrained(args.teacher_pretrained)
        hidden_size = int(args.teacher_pretrained.split('/')[1].split('_')[3].split('-')[-1])
        self.ff = nn.Linear(hidden_size, 1)

    def forward(self, ids, mask, token_type_ids):
        bert_output = self.model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = torch.sigmoid(self.ff(bert_output.pooler_output))
        return output


class TwinBERT(nn.Module):
    def __init__(self, args):
        super(TwinBERT, self).__init__()
        self.encoder_model = BertModel.from_pretrained(args.student_pretrained)

    def forward(self, seq1, mask1, seq2, mask2):
        output_1 = self.encoder_model(seq1, attention_mask=mask1).pooler_output
        output_2 = self.encoder_model(seq2, attention_mask=mask2).pooler_output
        cosine_similarity = nn.functional.cosine_similarity(output_1, output_2).unsqueeze(-1)
        return cosine_similarity


def data_collator(features):
    first = features[0]
    batch = {}

    batch["labels"] = torch.tensor([f["relevance_label"] for f in features], dtype=torch.float).unsqueeze(-1)
    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in features])
        elif isinstance(v, str):
            batch[k] = [f[k] for f in features]
        else:
            batch[k] = torch.tensor([f[k] for f in features])

    return batch


def _forward_pass(batch, args):
    score = None
    if args.model == 'teacher':
        score = model(batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device),
                      batch['token_type_ids'].to(args.device))
    elif args.model == 'student':
        score = model(batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device),
                      batch['input_ids_2'].to(args.device),
                      batch['attention_mask_2'].to(args.device))
    return score


def train(model, train_dataloader, val_dataloader, args):
    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = None

    for epoch in range(args.num_epochs):
        model.train()
        avg_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            label = batch['labels'].to(args.device)
            score = _forward_pass(batch, args)
            loss = nn.MSELoss()(score, label)

            loss.backward()
            optimizer.step()

            avg_loss += loss / args.logfreq
            if (step + 1) % args.logfreq == 0:
                logging.info('-- Epoch %s, Step: %s, Avg Loss = %.6f' % (epoch, step, avg_loss))
                avg_loss = 0.0

        # full validation after each epoch
        model.eval()
        labels, scores = [], []
        for batch in val_dataloader:
            with torch.no_grad():
                label, score = batch['labels'].to(args.device), _forward_pass(batch, args)
                labels.append(label)
                scores.append(score)

        labels, scores = torch.cat(labels, 0).view(-1, 1), torch.cat(scores, 0).view(-1, 1)
        val_loss = nn.MSELoss()(scores, labels)

        # calculate AUC
        labels, scores = labels.cpu(), scores.cpu()
        pr_auc, roc_auc = average_precision_score(labels, scores), roc_auc_score(labels, scores)
        logging.info('-- Epoch %s: PR AUC %.6f, ROC AUC %.6f, Validation Loss %.6f' % (epoch, pr_auc, roc_auc, val_loss))

        # save checkpoint
        torch.save(model, os.path.join(args.output_dir, 'model_%s.pth' % epoch))
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            logging.info('-- Best checkpoint so far!')
            torch.save(model, os.path.join(args.output_dir, 'model_best.pth'))


def predict(model, dataloader, args):
    model.eval()
    cnt = 0
    output_file = os.path.join(args.output_dir, 'prediction.tsv')
    logging.info('Write prediction results to %s.' % output_file)
    with open(output_file, 'w', encoding='utf-8') as fout:
        for batch in dataloader:
            with torch.no_grad():
                score = _forward_pass(batch, args)

            for i in range(len(score)):
                score_str = str(score.cpu()[i].numpy()[0])
                line = '\t'.join([batch['query'][i], batch['ad_title'][i], batch['ad_description'][i], score_str])
                fout.write(line + '\n')
                cnt += 1
                if cnt > 0 and cnt % 1000 == 0:
                    logging.info('-- Predict %s samples, done.' % cnt)


if __name__ == '__main__':
    # parse parameters
    args = parse_args()

    # set up device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set up dirs
    args.output_dir = os.path.join(args.output_dir, args.model)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # set up log file
    logfile = os.path.join(args.output_dir, 'log_%s.txt' % args.mode)
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='a+', format='%(asctime)-15s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('Arguments')
    for key, val in sorted(vars(args).items()):
        logging.info(f'-- Argument: {key}'.ljust(40) + f'-- {val}')

    # load dataset
    dataset = load_dataset('xglue', 'qadsm')
    if args.use_labeled_data == 'true':
        logging.info('Load labeled dataset.')
    else:
        dataset['train'] = load_dataset('kdd_2022_tutorial/load_dataset.py', split='train')
        logging.info('Load unlabeled dataset.')

    # load model and tokenizer
    mode, tokenizer = None, None
    if args.model == 'teacher':
        model = TeacherModel(args).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_pretrained)
        logging.info('Load Teacher model %s.' % args.teacher_pretrained)
    elif args.model == 'student':
        model = TwinBERT(args).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.student_pretrained)
        logging.info('Load Student model (TwinBERT) %s.' % args.student_pretrained)

    # load checkpoint
    if args.prev_ckpt is not None:
        model = torch.load(args.prev_ckpt).to(args.device)
        logging.info('Load model from %s' % args.prev_ckpt)


    def preprocess_function(examples):
        # Concatenate ad_title and ad_description
        texts = []
        for i in range(len(examples['query'])):
            new_text = (examples['query'][i], examples['ad_title'][i] + ' ' + examples['ad_description'][i])
            texts.append(new_text)

        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=False,
        )

    def preprocess_function_student(examples):
        # Concatenate ad_title and ad_description
        texts = []
        for i in range(len(examples['query'])):
            new_text = examples['ad_title'][i] + ' ' + examples['ad_description'][i]
            texts.append(new_text)

        tok_q = tokenizer(
            examples['query'],
            padding='max_length',
            truncation=True,
            max_length=args.max_length_query,
            return_special_tokens_mask=False,
        )

        tok_a = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=args.max_length_ad,
            return_special_tokens_mask=False,
        )
        tok_q['input_ids_2'] = tok_a['input_ids']
        tok_q['attention_mask_2'] = tok_a['attention_mask']
        tok_q['token_type_ids_2'] = tok_a['token_type_ids']
        return tok_q

    # process dataset
    tokenized_dataset = dataset.map(
        preprocess_function if args.model == 'teacher' else preprocess_function_student,
        batched=True,
        num_proc=1,
        # remove_columns=dataset["train"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset line_by_line",
    )

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size)
    val_dataloader = DataLoader(tokenized_dataset['test.en'], collate_fn=data_collator, batch_size=args.val_batch_size)

    if args.mode == 'train':
        logging.info('Training started.')
        train(model, train_dataloader, val_dataloader, args)
        logging.info('Training completed.')

    if args.mode == 'predict':
        logging.info('Inference started.')
        predict(model, train_dataloader, args)
        logging.info('Inference completed.')
