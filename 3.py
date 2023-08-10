#================微调chatglm模型.=========本地cpu版本!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.=========有时候加载失败,重新运行下, 可能是数据集里面的网络下载不好.
# 分3步. 经典的那个instructgpt图片.
# 第一个是train_sft, 第二个是train_rm      第三个是train_ppo
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)
from src.utils.config import (
    ModelArguments,
    DataTrainingArguments,
    FinetuningArguments,
    GeneratingArguments
)
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor

from peft.utils import CONFIG_NAME, WEIGHTS_NAME
#==================step 1===================================
from transformers.trainer import TRAINER_STATE_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME
def load_trainable_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    weights_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if os.path.exists(weights_file):  # 如果是整个文件的存储.
        model_state_dict = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False) # skip missing keys
    elif os.path.exists(os.path.join(checkpoint_dir, WEIGHTS_INDEX_NAME)): # 如果是分片模式的存储
        load_sharded_checkpoint(model, checkpoint_dir, strict=False)
    else:
        print("Provided path ({}) does not contain pre-trained weights.".format(checkpoint_dir))
        return False
    return True


import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1'
IGNORE_INDEX = -100
VALUE_HEAD_FILE_NAME = "value_head.bin"
FINETUNING_ARGS_NAME = "finetuning_args.json"

def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
    if not os.path.exists(valuehead_file):
        print("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])      # 需要学的参数, 注册进model
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
    model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))
    return True

# coding=utf-8
# Implements several parameter-efficient supervised fine-tuning method for ChatGLM.
# This code is inspired by https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py

#模型进行peft化
from typing import Dict, List, Optional

import torch
def prepare_model_for_training(
        model: PreTrainedModel,
        finetuning_type: str,
        output_embedding_base_layer: torch.nn.Module,
        output_embedding_layer_name: Optional[str] = "lm_head",
        use_gradient_checkpointing: Optional[bool] = True,
        layer_norm_names: Optional[List[str]] = ["layernorm"] # for chatglm setting
) -> PreTrainedModel:

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if finetuning_type != "full" and hasattr(output_embedding_base_layer, output_embedding_layer_name):
        output_embedding_layer = getattr(output_embedding_base_layer, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(output_embedding_base_layer, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))#设置模型输出为float32.

    return model


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
                trainable_params, all_param, 100 * trainable_params / all_param))
def init_adapter(
        model: PreTrainedModel,   #模型.
        model_args: ModelArguments,  # 模型的一些配置参数.
        finetuning_args: FinetuningArguments,# 输入4种输入模式.
        is_trainable: bool
) -> PreTrainedModel:
    r"""
    Initializes the adapters.

    Support full-parameter, freeze, P-Tuning v2 and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable: # 训练的话, finetune模式一定要选一个.
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full":
        print("Fine-tuning method: Full")
        model = model.float() # 参数都float化.

    if finetuning_args.finetuning_type == "freeze":
        print("Fine-tuning method: Freeze")
        #先锁参数.
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in finetuning_args.trainable_layers): # 如果参数的名字name不包含训练层中的任何一个,那么不就是说他应该被锁定,所以我们梯度false.
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)  # 否则就是要训练.训练的话我们都f32化. 从这里也看出来非float的是不能训练的. 训练只支持小数模式.
        #加载参数.
        if model_args.checkpoint_dir is not None:
            assert load_trainable_params(model, model_args.checkpoint_dir[0]), "Model checkpoint is not correctly loaded." # 进行模型加载.

    if finetuning_args.finetuning_type == "p_tuning":
        print("Fine-tuning method: P-Tuning v2")

        if model_args.checkpoint_dir is not None:
            assert load_trainable_params(model, model_args.checkpoint_dir[0]), "Model checkpoint is not correctly loaded." # ptuning也是先加载全部参数.

    if finetuning_args.finetuning_type == "lora":
        print('loara')
        lastest_checkpoint = None

        if model_args.checkpoint_dir is not None: # 如果已经存在checkpoint
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], WEIGHTS_NAME)), \
                "Provided path ({}) does not contain a LoRA weight.".format(model_args.checkpoint_dir[0])
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], CONFIG_NAME)), \
                "The given checkpoint may be not a LoRA checkpoint, please specify `--finetuning_type full/p_tuning/freeze` instead."
#lora也需要有config
            if is_trainable and model_args.resume_lora_training: # continually train on the lora weights
                checkpoints_to_merge, lastest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir
            #模型peft化.
            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                print(1)

            if lastest_checkpoint is not None: # resume lora training
                model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=True)

        if is_trainable and lastest_checkpoint is None: # create new lora weights while training
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, # we should regard ChatGLM as a causal LM
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        print(1)

    return model


import os
os.environ["WANDB_DISABLED"] = "true"
from src.utils import (
    DataCollatorForChatGLM,
    Seq2SeqTrainerForChatGLM,
    ComputeMetrics,
    LogCallback,
    load_pretrained,
    prepare_args,
    prepare_data,
    preprocess_data,
    get_logits_processor,
    plot_loss
)

if 1:
    # Prepare pretrained model and dataset
    import sys
    # print(sys.argv)
    sys.argv="train_sft.py  --do_train  \
    --dataset alpaca_gpt4_zh \
    --finetuning_type lora \
    --output_dir path_to_sft_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --use_v2  \
    ".split( ) #=========这个变量是各个脚本通用的.直接修改即可.
    model_args, data_args, training_args, finetuning_args = prepare_args(stage="sft")
#     model_args.model_name_or_path='gpt2'  #=========改成一个小模型进行学习.
#     model_args.use_v2=False  #=========改成一个小模型进行学习.
#     print(model_args,11111111111111)
    # print(data_args)
#     print(training_args)
#     print(finetuning_args)
    dataset = prepare_data(model_args, data_args)












    
    # model, tokenizer = load_pretrained(model_args, finetuning_args, training_args.do_train, stage="sft")

    stage="sft"
    is_trainable=training_args.do_train
    if (not is_trainable) and model_args.checkpoint_dir is None:

        finetuning_args = FinetuningArguments(finetuning_type="none")
    # 后2种stage只能用lora. 因为不是lm模型.
    assert stage == "sft" or finetuning_args.finetuning_type == "lora", \
        "RM and PPO training can only be performed with LoRA method."

    quantization = None
    if model_args.quantization_bit is not None:
        if is_trainable:
            if finetuning_args.finetuning_type == "full":
                raise ValueError("Full-parameter fine-tuning does not support quantization.")
            elif finetuning_args.finetuning_type == "p_tuning":
                quantization = "cpm" # use cpm's quantization
            else:
                quantization = "bnb" # use bnb's quantization
        else:
            quantization = "cpm"     # 量化方法: cpm 和bnb  后续看看如何实现.

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
        **config_kwargs
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        **config_kwargs
    )

    # P-Tuning v2 configurations. Use the built-in p-tuning method of ChatGLM.
    if finetuning_args.finetuning_type == "p_tuning":
        config.pre_seq_len = finetuning_args.pre_seq_len # enable this will fix other parameters automatically
        config.prefix_projection = finetuning_args.prefix_projection

    # Quantization configurations for Full, Freeze and LoRA in training (using bitsandbytes library).
    if quantization == "bnb":#使用bitsandbytes模式量化.
        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("transformers>=4.30.1", "To fix: pip install transformers>=4.30.1")
            require_version("accelerate>=0.20.3", "To fix: pip install accelerate>=0.20.3")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if model_args.checkpoint_dir is not None and finetuning_args.finetuning_type == "full":
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    # Load and prepare pretrained models (without valuehead).
    print('打印加载参数',model_to_load,config,config_kwargs)
    model = AutoModel.from_pretrained(model_to_load, config=config, **config_kwargs).half().cuda()

    # Register auto class to save the custom code files.
    if hasattr(config, "auto_map") and "AutoConfig" in config.auto_map:
        config.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
        tokenizer.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoModel" in config.auto_map:
        model.__class__.register_for_auto_class()

    if model_args.use_v2: # 使用v2版本.
        assert tokenizer.eos_token_id is not None, "Please update the *.json and *.py files of ChatGLM2-6B from HuggingFace."
        model.lm_head = model.transformer.output_layer
        output_embedding_base_layer = model.transformer
        output_embedding_layer_name = "output_layer"
    else:
        assert tokenizer.eos_token_id == 130005, "Please specify `use_v2` argument while using ChatGLM2-6B."
        output_embedding_base_layer = model
        output_embedding_layer_name = "lm_head"

    # Initialize adapters
    model = prepare_model_for_training(
        model,
        finetuning_args.finetuning_type,
        output_embedding_base_layer,
        output_embedding_layer_name
    ) if is_trainable else model
    model = init_adapter(model, model_args, finetuning_args, is_trainable)

    if not is_trainable:#非训练模式.
        model.requires_grad_(False) # fix all model params
        model = model.half() # cast all params to float16 for inference

    # Quantization with the built-in method for P-Tuning v2 training or evaluation.
    # Model parameters should be cast to float16 in quantized P-Tuning setting.
    if quantization == "cpm":
        if is_trainable: # convert all params into half precision except prefix_encoder in training
            for name, param in model.named_parameters():
                if "prefix_encoder" not in name:
                    param.data = param.data.to(torch.float16) # 非训练的部分都设置为f16.

        model.quantize(model_args.quantization_bit) # built-in method in ChatGLM-6B, also an in-place operation
    from peft.utils import CONFIG_NAME, WEIGHTS_NAME

    from trl import AutoModelForCausalLMWithValueHead

    if quantization is not None:
        print("Quantized model to {} bit.".format(model_args.quantization_bit))
    # 非lm模型.加一个value头.
    if stage == "rm" or stage == "ppo": # add value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        if stage == "rm" and model_args.checkpoint_dir is not None: # load valuehead weights to evaluate reward model # 读取旧的权重.
            print("Only the last checkpoint containing valuehead will be loaded as the valuehead.")
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                #如果导入模型成功.
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo": # load reward model
            assert is_trainable, "PPO stage cannot be performed at evaluation."
            assert model_args.reward_model is not None, "Reward model is necessary for PPO training."
            print("Load reward model from {}".format(model_args.reward_model))
            model.pretrained_model.load_adapter(model_args.reward_model, "reward", is_trainable=False)
            assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."

    print_trainable_params(model)















    # model=model.float() #======因为默认参数不是float化的.###############如果是显卡的话关闭这个.
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="sft")
    data_collator = DataCollatorForChatGLM(
        tokenizer=tokenizer,
        model=model,
        ignore_pad_token_for_loss=(data_args.ignore_pad_token_for_loss and not training_args.predict_with_generate),
        use_v2=model_args.use_v2
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length if \
                training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.eval_num_beams if \
                data_args.eval_num_beams is not None else training_args.generation_num_beams

    # Split the dataset
    if training_args.do_train:
        if data_args.dev_ratio > 1e-6:
            dataset = dataset.train_test_split(test_size=data_args.dev_ratio)
            trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            trainer_kwargs = {"train_dataset": dataset}
    else: # do_eval or do_predict
        trainer_kwargs = {"eval_dataset": dataset}

    # Initialize our Trainer
    trainer = Seq2SeqTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **trainer_kwargs
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = {
        "do_sample": True,
        "top_p": 0.7,
        "max_new_tokens": data_args.max_target_length + 1,
        "temperature": 0.95,
        "logits_processor": get_logits_processor()
    }

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

import transformers
from transformers import  AutoModel,AutoTokenizer,AutoConfig,DataCollatorForSeq2Seq
from peft import get_peft_model, AdaLoraConfig, TaskType

peft_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
    target_modules=["query", "value",'query_key_value']
)


# target_modules, the name of the attention matrices to apply LoRA to (q_proj and v_proj, or query and value in this case)