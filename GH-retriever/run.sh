for seed in 0 1 2 3  
do
# 1) inference only
# a) Question-Only
#python inference.py --dataset expla_graphs --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0 --seed $seed
#python inference.py --dataset webqsp --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0 --seed $seed


# 2) frozen llm + prompt tuning
# a) prompt tuning
#python train.py --dataset expla_graphs --model_name pt_llm --seed $seed
#python train.py --dataset scene_graphs --model_name pt_llm --seed $seed

# b) g-retriever
#python train.py --dataset expla_graphs --model_name graph_llm --seed $seed
python train.py --dataset scene_graphs --model_name graph_llm --seed $seed --batch_size 3
#python train.py --dataset webqsp --model_name graph_llm --seed $seed

# 3) tuned llm
# a) finetuning with lora
#python train.py --dataset expla_graphs --model_name llm --llm_frozen False --seed $seed --batch_size 4
#python train.py --dataset scene_graphs_baseline --model_name llm --llm_frozen False --seed $seed --batch_size 2
#python train.py --dataset webqsp_baseline --model_name llm --llm_frozen False --seed $seed --batch_size 4
# b) g-retriever + finetuning with lora
#python train.py --dataset expla_graphs --model_name graph_llm --llm_frozen False --seed $seed --batch_size 4
#python train.py --dataset scene_graphs --model_name graph_llm --llm_frozen False --seed $seed --batch_size 3
#python train.py --dataset webqsp --model_name graph_llm --llm_frozen False --seed $seed --batch_size 4


#python train.py --dataset kg --model_name graph_llm --seed $seed 



done

#git add .
#git commit -m "Run script updated for GH-retriever"
#git push origin main