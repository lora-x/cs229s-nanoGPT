import json
import matplotlib.pyplot as plt
import os

batch_sizes = [1, 2, 4, 8, 16, 32]
shakespeare_file_names = [f"out/shakespeare/results_gpt2-medium_shakespeare_batch{i}_eval_iters50_ft_iters20" for i in batch_sizes]
wikitext_file_names = [f"out/wikitext/results_gpt2-medium_wikitext_batch{i}_eval_iters50_ft_iters20" for i in batch_sizes]

# Lists to store the extracted data for both datasets
shakespeare_memory, shakespeare_throughput, wikitext_memory, wikitext_throughput = [], [], [], []

def read_and_extract_data(file_name, field):
    with open(file_name, 'r') as file:
        data = json.load(file)
        value = data[field]
        return value

def fill_list(file_names, memory_list, throughput_list):
    for file_name in file_names:
        memory = read_and_extract_data("max_memory_per_gpu")
        memory_list.append(memory)
        
        tokens_per_iter = read_and_extract_data("tokens_per_iter")
        average_time_per_iter = read_and_extract_data("average_time_per_iter")
        throughput = tokens_per_iter / average_time_per_iter
        throughput_list.append(throughput)
        

fill_list(shakespeare_file_names, shakespeare_memory, shakespeare_throughput)
fill_list(wikitext_file_names, wikitext_memory, wikitext_throughput)

# Create a scatter plot of max_memory_per_gpu vs batch_size for both datasets
plt.figure()
plt.scatter(batch_sizes, shakespeare_memory, color='orange', label='Shakespeare')
plt.scatter(batch_sizes, wikitext_memory, color='blue', label='Wikitext')
plt.xlabel('Batch Size')
plt.ylabel('Max Memory per GPU (bytes)')
plt.title('Memory Usage vs Batch Size')
plt.legend()
plt.save('out/memory_usage_vs_batch_size.png')

plt.figure()
plt.scatter(batch_sizes, shakespeare_throughput, color='orange', label='Shakespeare')
plt.scatter(batch_sizes, wikitext_throughput, color='blue', label='Wikitext')
plt.xlabel('Batch Size')
plt.ylabel('Throughput (tokens/second)')
plt.title('Throughput vs Batch Size')
plt.legend()
plt.save('out/throughput_vs_batch_size.png')
