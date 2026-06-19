# What I Need to Learn

## Deploy vLLM: Set up and run the vLLM system from its GitHub repository.  

o Clone the code and be familiar with how to make changes and recompile.  
- vLLM gitHub URL: https://github.com/vllm-project/vllm
## Measure latency and throughput across phases (prefill and decode).  

o Instrument the system to capture performance metrics for latency  (end-to-end and per-phase) and throughput.  

- What is prefill and decode (Day 5 & 6)

o Focus on measuring the time and resource usage during the prefill  phase (e.g., initial processing, input handling) and the decode phase  (actual model inference/generation).  
- Practice on bench and analyze bottleneck (Day 10, 12, & 17)
## Analyze how different factors—datasets, request arrival rate, model  type, GPU count, and parallelism—affect system performance.  

§ Datasets: Choose and document at least two different datasets that simulate realistic input workloads.  
- Measure and compare realistic workload and synthesis (Day11 & 14)

§ Simulate different load conditions by varying the arrival rate of requests.  
-  change request rate (Day 12)
 
§ Deploy at least two models (or different model sizes) available within the vLLM framework. 
   
§ Test performance using varying numbers of GPUs.  
- Concurrency and GPUs affection
  
§ Document CPU performance if applicable.  
  
§ Enable and evaluate parallel processing options within vLLM.  
  
## Analyze results to determine performance bottlenecks.  
## Generate figures to clearly present and compare the performance metrics.

# GPU Comparison
### RunPod
URL: https://www.runpod.io/product/cloud-gpus

### Google Colab
### Vast,ai