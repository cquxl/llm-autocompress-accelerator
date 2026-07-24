
import os
import torch
from lm_eval.utils import make_table
import shutil

from cfg import get_args
from data import get_dataloader
from model import get_model
from prune import D2Prune, Pruner
from utils import eval_ppl, eval_zero_shot, evaluate, eval_lmm_zero_shot
import sys
import torch
import os
os.environ["HF_ALLOW_CODE_EXECUTION"] = "1"
os.environ["TRANSFORMERS_DISABLE_SAFE_LOADING"] = "1"

# print("Torch CUDA:", torch.version.cuda)
# print("Torch cuDNN:", torch.backends.cudnn.versio
#-----------------------------------loading args from parameters yaml file----------------------------------------------#
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

cfg_path = './cfg/model.yaml'
args = get_args(cfg_path)

#-----------------------------------loading model and tokenizer---------------------------------------------------------
if args.free:
    model, tokenizer = get_model(args.model, device_type="cpu", seq_len=args.seqlen) # cpu loading
else:
    model, tokenizer = get_model(args.model, device_type="auto", seq_len=args.seqlen) # gpu loading

if "llava" in args.model.lower():
    processor = tokenizer
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

model.eval()
args.seq_len = model.seq_len
model_name = args.model.split("/")[-1]
args.model_name = model_name

def main(demo=False):
    if args.sparsity_ratio != 0:
        # loading calibration dataloader
        if 'deit' in args.model.lower():
            args.cali_data_path = args.root_data_path
            args.is_train = True
        train_loader = get_dataloader(args, tokenizer, model.seq_len,
                                      args.cali_data_path, eval_mode=False)


        args.logger.info("pruning starts")
        if args.prune_method == 'd2prune':
            pruner = D2Prune(args, model).pruner
        else:
            if args.prune_method == 'pruner-zero':
                from prune.pruner_zero import get_layer_gradient, GPTree
                args.gradients_l2 = get_layer_gradient(args, model, train_loader, args.device, data_cache_dir='./cache')
                args.engine = GPTree.load_tree('./prune/pruner_zero/best_tree.json')
            pruner = Pruner(args, model).pruner
        # pruner.prune_llm(train_loader)
        if 'deit' in args.model.lower():
            pruner.prune_vit(train_loader)
        else:
            pruner.prune_llm(train_loader)

        # if pruner.save_attention_score or pruner.save_activations:
        #     sys.exit() #  below not running

        # Save the pruned model
        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
            args.logger.info(f"save model to {args.save_model}")

    args.save_filepath = os.path.join(args.output_dir, f"log_{args.prune_method}.txt")

    # for llava-->evaluate ['scienceqa', 'MMBench']
    if 'llava' in args.model.lower():
        # save to a directory, then using model path to evaluate
        # args.sparsity_ratio = 0.7 # test
        # if args.sparsity_ratio >= 0.7:
        #     save_dir = os.path.join(args.output_dir, 'pruned_model')
        #     model.save_pretrained(save_dir)
        #     processor.save_pretrained(save_dir)

        # using prune_model path to evaluate
        # model = save_dir
        # model.to(args.device)
        # model.language_model.to(args.device)
        task_list = ["scienceqa_img", "mmbench_en"]
        # if args.sparsity_ratio >= 0.7: # 0.7 runing so long , try to evaluate using model path ( i don't know the reason presently)
        #     args.logger.info(f"evaluate by hf model path")
        #     results = eval_lmm_zero_shot(args, save_dir, task_list=task_list)
        # else:
        #     args.logger.info(f"evaluate by hf model")
        #     model.to(args.device)
        #     results = eval_lmm_zero_shot(args, model, task_list=task_list)
        if args.eval_zero_shot:
            results = eval_lmm_zero_shot(args, model, task_list=task_list)
            # 4.2 保存只包含 metrics 的精简版
            metrics_only = {
                "results": results.get("results", {}),
                "groups": results.get("groups", {}),
                "n-samples": results.get("n-samples", {}),
            }
            args.logger.info(metrics_only)
            try:
                args.logger.info("\n" + make_table(metrics_only))
            except:
                pass
        # del pruned_model dir
        # if os.path.exists(save_dir):
        #     shutil.rmtree(save_dir)
        sys.exit()

    # loading eval dataloader
    if 'deit' in args.model.lower():
        args.eval_data_path = args.root_data_path
        args.is_train = False
        test_loader = get_dataloader(args, tokenizer, model.seq_len, args.eval_data_path, eval_mode=True)
        acc1_list, acc5_list = evaluate(test_loader, model, args.device, use_amp=False)
        args.logger.info(f"global mean ACC1:{sum(acc1_list) / len(acc1_list)}")
        args.logger.info(f"global mean ACC5:{sum(acc5_list) / len(acc5_list)}")
        sys.exit()
    eval_loader = get_dataloader(args, tokenizer, model.seq_len, args.eval_data_path, eval_mode=True) # Tuple[Tensor,Tensor]
    test_loader = torch.stack([loader[0] for loader in eval_loader]) # [n,1,2048]->Tensor

    # ppl test
    if not args.free:
        ppl_test = eval_ppl(args, model, test_loader)
    ## device offloading
    else:
        ppl_test = eval_ppl(args, model, test_loader, is_split=True)
    # ppl_test = eval_ppl(args, model, test_loader, is_split=True)

    # zero-shot acc test
    if args.eval_zero_shot:
        task_list = None
        if demo:
            # task_list = ['boolq']
            task_list = ['mmlu']

        results = eval_zero_shot(args, model, tokenizer, task_list=task_list)
        args.logger.info("\n" + make_table(results))
        with open(args.save_filepath, "w") as f:
            if ppl_test:
                print("********************************")
                print("method\tsparsity\tppl_test", file=f, flush=True)
                print(f"{args.prune_method}\t{args.sparsity_ratio}\t{ppl_test:.4f}", file=f,
                      flush=True)
            print("********************************")
            print("zero_shot_results", file=f, flush=True)
            print(make_table(results), file=f, flush=True)
        args.logger.info(f"save filepath:{args.save_filepath}")


if __name__ == "__main__":
    main()