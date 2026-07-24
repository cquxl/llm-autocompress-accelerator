from .data_load import PruneDataset, ImageNet


def get_dataloader(args, tokenizer=None, seq_len=None, data_path=None, eval_mode=False):
    if 'deit' in args.model.lower():
        if eval_mode:
            prune_dataset = ImageNet(args,  dataset_name=args.cali_dataset, local_path=data_path, eval_mode=True)
        else:
            prune_dataset = ImageNet(args,  dataset_name=args.eval_dataset, local_path=data_path, eval_mode=False)
        return prune_dataset.dataloader

    if eval_mode:
        prune_dataset = PruneDataset(args, args.eval_dataset, tokenizer, seq_len, data_path, eval_mode=True)
    else:
        prune_dataset = PruneDataset(args, args.cali_dataset, tokenizer, seq_len, data_path, eval_mode=False)
    return prune_dataset.dataloader