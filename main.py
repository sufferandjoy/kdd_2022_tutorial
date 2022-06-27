# This is the entry script for KDD tutorial

import os
import logging
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score, roc_auc_score
from model import TeacherModel, TwinBERT
from utils import prepare_dataset, prepare_data_loader, _forward_pass, print_eval_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Params")

    # overall config
    parser.add_argument("--task", type=str, default='eval', help="eval/teacher_ft/student_ft/teacher_inf/student_kd.")
    parser.add_argument("--load_dataset_py_path", type=str, default='load_dataset.py', help="Path to load_dataset.py.")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Batch size (per device) for the training.")
    parser.add_argument("--val_batch_size", type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--logfreq", type=int, default=100, help="Frequency to print training logs.")
    parser.add_argument("--output_dir", type=str, default='output', help="Where to store the final model.")

    # config on teacher model
    parser.add_argument("--teacher_pretrained", type=str, default='google/bert_uncased_L-4_H-256_A-4', help="Path to pretrained model for teacher.")
    parser.add_argument("--max_length", type=int, default=32, help="Maximum sequence length after tokenization.")

    # config on student model
    parser.add_argument("--student_pretrained", type=str, default='google/bert_uncased_L-2_H-128_A-2', help="Path to pretrained model for TwinBERT encoder")
    parser.add_argument("--max_length_query", type=int, default=9, help="Maximum query sequence length after tokenization.")
    parser.add_argument("--max_length_ad", type=int, default=24, help="Maximum ad sequence length after tokenization.")

    args = parser.parse_args()
    return args

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
            score = _forward_pass(model, batch, args)
            loss = nn.MSELoss()(score, label)

            loss.backward()
            optimizer.step()

            avg_loss += loss / args.logfreq
            if (step + 1) % args.logfreq == 0:
                logging.info('-- Epoch %s, Step: %s, Avg Loss = %.6f' % (epoch, step, avg_loss))
                avg_loss = 0.0

        # full validation after each epoch
        metrics = eval(model, val_dataloader, args)
        pr_auc, roc_auc, val_loss = metrics['pr_auc'], metrics['roc_auc'], metrics['val_loss']
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
                score = _forward_pass(model, batch, args)

            for i in range(len(score)):
                score_str = str(score.cpu()[i].numpy()[0])
                line = '\t'.join([batch['query'][i], batch['ad_title'][i], batch['ad_description'][i], score_str])
                fout.write(line + '\n')
                cnt += 1
                if cnt > 0 and cnt % 1000 == 0:
                    logging.info('-- Predict %s samples, done.' % cnt)


def eval(model, dataloader, args):
    model.eval()
    labels, scores = [], []
    for batch in dataloader:
        with torch.no_grad():
            label, score = batch['labels'].to(args.device), _forward_pass(model, batch, args)
            labels.append(label)
            scores.append(score)

    labels, scores = torch.cat(labels, 0).view(-1, 1), torch.cat(scores, 0).view(-1, 1)
    val_loss = nn.MSELoss()(scores, labels)

    # calculate evaluation metrics
    metrics = {}
    labels, scores = labels.cpu(), scores.cpu()
    pr_auc, roc_auc = average_precision_score(labels, scores), roc_auc_score(labels, scores)

    metrics['pr_auc'] = pr_auc
    metrics['roc_auc'] = roc_auc
    metrics['val_loss'] = val_loss

    return metrics


if __name__ == '__main__':
    # parse parameters
    args = parse_args()

    # set up device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set up dirs
    args.output_dir = os.path.join(args.output_dir, args.task)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # set up log file
    logfile = os.path.join(args.output_dir, 'log.%s.txt' % args.task)
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='a+', format='%(asctime)-15s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('Arguments')
    for key, val in sorted(vars(args).items()):
        logging.info(f'-- Argument: {key}'.ljust(40) + f'-- {val}')

    if args.task != 'eval':
        # train or inference on a single model
        args.model, args.mode = args.task.split('_')

        # load dataset and tokenizer
        dataset = prepare_dataset(args)
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_pretrained)

        # load model and tokenizer
        model, tokenizer = None, None
        if args.model == 'teacher':
            model = TeacherModel(args).to(args.device)
            logging.info('Load Teacher model %s.' % args.teacher_pretrained)
        elif args.model == 'student':
            model = TwinBERT(args).to(args.device)
            logging.info('Load Student model (TwinBERT) %s.' % args.student_pretrained)

        train_dataloader, val_dataloader = prepare_data_loader(dataset, tokenizer, args)

        if args.mode == 'ft' or args.mode == 'kd':
            logging.info('Training started.')
            train(model, train_dataloader, val_dataloader, args)
            logging.info('Training completed.')

        if args.mode == 'inf':
            ckpt_path = 'output/%s_ft/model_best.pth' % args.model
            model = torch.load(ckpt_path).to(args.device)
            logging.info('Load model from %s' % ckpt_path)

            logging.info('Inference started.')
            predict(model, train_dataloader, args)
            logging.info('Inference completed.')
    else:
        # conduct full evaluation
        tasks = ['teacher_ft', 'student_ft', 'student_kd']

        # load dataset and tokenizer
        args.mode = 'ft'
        dataset = prepare_dataset(args, splits_to_keep='test.en')
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_pretrained)

        res = {}
        for task in tasks:
            logging.info('Evaluate Task %s.' % task)
            args.model = task.split('_')[0]

            # load model
            ckpt_path = 'output/%s/model_best.pth' % task
            model = torch.load(ckpt_path).to(args.device)
            logging.info('-- Load model from %s' % ckpt_path)

            _, val_dataloader = prepare_data_loader(dataset, tokenizer, args, use_train=False)
            res[task] = eval(model, val_dataloader, args)

        # compare results
        print_eval_metrics(res, ['PR AUC', 'ROC AUC'])
