from langchain.llms import LlamaCpp

model = LlamaCpp(model_path="/work/u9796576/Llama/quantized_q4_1.gguf",
               n_gpu_layers=-1,#If -1, all layers are offloaded.
               n_batch=256,
               n_ctx=0, #0 = from model
               f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
               verbose=False,
               temperature=0.1,
               top_k=50,
               top_p=0.95,
               stop=['\n', 'ENDINPUT','ENDINSTRUCTION'])