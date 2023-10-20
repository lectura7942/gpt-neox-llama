def print_HF_config(model_id:str):
    """HF Config 출력"""
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_id)
    print(config)

def tmp():
    import jsonlines
    
    cnt = 0
    with jsonlines.open("/home/ubuntu/gpt-neox-llama/data/news_data_2gb.jsonl") as f:
        try:
            for line in f.iter():
                cnt += 1
        except:
            print(f"error at {cnt}")
            pass
    print(f"There are {cnt} data.")

def test_infer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from megatron.tokenizer.tokenizer import HFLlamaTokenizer
    from tqdm import tqdm

    model_id = "hf_model"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=True, add_eos_token=True) # default don't add eos
    # eod tokens added in training
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # load_in_8bit = True,
        # device_map="auto", # need accelerate
        torch_dtype=torch.float16,
    )
    # TODO add pad id. Currently not using
    
    # model.config.pad_token_id = tokenizer.pad_id
    
    eval_dataset = [
        "On Saturday, William Shatner took to Twitter to mourn",
        "Speculations about the First Lady's",
        "This evening",
    ]

    results = []

    device = torch.device("cuda:0")

    model.to(device).eval()

    for idx, d in tqdm(enumerate(eval_dataset), desc="Generating", total=len(eval_dataset)):
        prompt = d
        model_input = tokenizer(
            prompt, return_token_type_ids=False,
            return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_text = tokenizer.decode(
                model.generate(**model_input, max_new_tokens=100)[0], 
                # skip_special_tokens=True
            )

        results.append(output_text)
    
    for r in results: print(r)



if __name__ == "__main__":
    # print_HF_config("EleutherAI/gpt-neo-125m")
    # tmp()
    
    test_infer()
    