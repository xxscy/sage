from sage.testing.rag_cot_chain import RAGCOTConfig
from langchain.schema import HumanMessage

cfg = RAGCOTConfig()
if hasattr(cfg.llm_config, 'streaming') and cfg.llm_config.streaming:
    cfg.llm_config.streaming = False

llm = cfg.llm_config.instantiate()
print('Sending test prompt...')
resp = llm([HumanMessage(content='Connectivity probe. Reply with OK only.')])
print('Response:', resp)
