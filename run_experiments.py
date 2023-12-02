import json

'''
To get validation loss and training throughput, run:
    torchrun --standalone --nproc_per_node 4 --nnodes 1 train_fsdp.py config/train_wikitext.py
To make inference, run:
    python sample --out_dir=wikitext
'''
# ----------------------------GET_RESULTS------------------------------------
results_dict = {
    "loss": 2.86,
    "inference_latency_1": 100.0, # TODO: haven't got a value yet
    "inference_latency_12": 100.0, # TODO: haven't got a value yet
    "training_throughput_4": 3832.0,
    "training_throughput_12": 0.0 # TODO: haven't got a value yet
}

# ----------------------------WRITE_TO_JSON_FILE-----------------------------------
# Writing to sample.json
with open("results.json", "w") as outfile:
    json.dump(results_dict, outfile)