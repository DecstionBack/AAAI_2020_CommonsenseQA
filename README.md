# Paper
Graph-Based Reasoning over Heterogeneous External Knowledge for Commonsense Question Answering. (published at AAAI 2020)
# Environments
- torch>=1.0
- python >=3.0

# Data download
You can download the [AAAI2020-data.tar.gz](https://1drv.ms/u/s!AlDIAan0xkG1oEstTFs45BD2JQZA?e=KQ8Pok) file used in this paper and extract it as `data` folder.

# Training, evaluation and test
```
CUDA_VISIBLE_DEVICES=0,1 python run.py \
--model_type xlnet \
--model_name_or_path xlnet-large-cased \
--do_test \
--do_train \
--do_eval \
--data_dir data \
--output_dir xlnet_large_commonsenseQA \
--max_seq_length 256 \
--eval_steps 200 \
--per_gpu_train_batch_size 2 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 4 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 40000 \
--report_steps 1000
```
Our code refers to the pytorch_transformers project supported by huggingface. We run experiments on 2 P100 GPUs. The training process may consume about 50 GB RAM. The XLNet large model will be downloaded automatically. The code will train the model and evaluate on the development dataset every `--eval_steps`. After training, it will output the predicted results on the test dataset. You can change `--train_steps` to reduce the training time.


# Citation
If you refer to our code or our paper, please cite our paper as follows:
```
@inproceedings{lv2020commonsense,
  author    = {Shangwen Lv,   Daya Guo, Jingjing Xu, Duyu Tang, Nan Duan, Ming Gong, Linjun Shou, Daxin Jiang, Guihong Cao and Songlin Hu},
  title     = {Graph-Based Reasoning over Heterogeneous External Knowledge for Commonsense Question Answering},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2020, New York, USA},
  pages     = {8449--8456},
  publisher = {{AAAI} Press},
  year      = {2020}
}
```
