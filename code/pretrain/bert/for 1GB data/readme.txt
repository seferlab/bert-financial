-- first save a tokenizer model using bert_tokenizer.ipynb file
-- then, download transformers 
	git clone https://github.com/huggingface/transformers.git
	also,
	pip install datasets

-- then, go to /transformers/examples/pytorch/language-modeling/run_mlm.py file
-- change around line 293
	    if model_args.tokenizer_name:
        	#tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        	from transformers import BertTokenizerFast, RobertaTokenizerFast
        	#for bert uncomment
        	tokenizer = BertTokenizerFast.from_pretrained("./bert", max_len=512)
        	#for roberta uncomment
        	#tokenizer = RobertaTokenizerFast.from_pretrained("./robertam", max_len=512)

-- then,
nohup python -u /home/emresefer/zehra_old/1GBdata/transformers/examples/pytorch/language-modeling/run_mlm.py --train_file "/home/emresefer/zehra_old/1GBdata/out2006.txt" --tokenizer_name "/content/xlnet" --do_train --max_steps 1000 --model_type bert --pad_to_max_length True --output_dir ./output_bert
