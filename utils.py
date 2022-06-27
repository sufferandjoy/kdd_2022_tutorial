import logging
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


def prepare_dataset(args, splits_to_keep='train,test.en'):
    dataset = load_dataset('xglue', 'qadsm')
    if args.mode == 'kd':
        dataset['train'] = load_dataset(args.load_dataset_py_path, split='train')
        logging.info('Load unlabeled dataset using %s.' % args.load_dataset_py_path)
    else:
        logging.info('Load labeled dataset.')

    splits_to_keep = splits_to_keep.split(',')
    for key in dataset.copy():
        if key not in splits_to_keep:
            dataset.pop(key)

    dataset = dataset.filter(lambda example: not (
                example['ad_title'].startswith('ERROR_AdRejected') or example['ad_description'].startswith(
            'ERROR_AdRejected')))

    return dataset


def _data_collator(features):
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


def prepare_data_loader(dataset, tokenizer, args, use_train=True):

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
        # desc="Running tokenizer on dataset line_by_line",
    )
    train_dataloader = None
    if use_train:
        train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, collate_fn=_data_collator, batch_size=args.train_batch_size)
    val_dataloader = DataLoader(tokenized_dataset['test.en'], collate_fn=_data_collator, batch_size=args.val_batch_size)

    return train_dataloader, val_dataloader


def _forward_pass(model, batch, args):
    score = None
    if args.model == 'teacher':
        score = model(batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device),
                      batch['token_type_ids'].to(args.device))
    elif args.model == 'student':
        score = model(batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device),
                      batch['input_ids_2'].to(args.device),
                      batch['attention_mask_2'].to(args.device))
    return score


def print_eval_metrics(results, metrics_to_print, control='student_ft', treatment='student_kd'):
    col_width = 10
    header = f'Task'.ljust(col_width)
    for m in metrics_to_print:
        header += f'{m}'.rjust(col_width)
    logging.info(header)

    for task in results:
        line = f'{task}'.ljust(col_width)
        for m in metrics_to_print:
            key = m.lower().replace(' ', '_')
            val = results[task][key]
            line += f'{val:.4f}'.rjust(col_width)
        logging.info(line)

    # print delta for each metric
    logging.info('-' * (col_width * (1+len(metrics_to_print)) + 2))
    line = f'delta'.ljust(col_width)
    for m in metrics_to_print:
        key = m.lower().replace(' ', '_')
        val = (results[treatment][key] / results[control][key] - 1) * 100
        line += f'{val:.2f}%'.rjust(col_width)
    logging.info(line)
