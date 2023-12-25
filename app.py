import os
from langchain.llms import CTransformers
from langchain import PromptTemplate,LLMChain
import chainlit as cl

local_llm = 'mistral-7b-openorca.Q2_K.gguf'

config = {
  'max_new_tokens':1024,
  'repetition_penalty':1.1,
  'temperature':0.5,
  'top_k':50,
  'top_p':0.9,
  'stream':True,
  'threads': int(os.cpu_count()/2)
}

llm_init = CTransformers(
  model = local_llm,
  model_type = "mistral",
  **config)
print(llm_init)

template = """Question: {question}
Answer: Let's think step by step and answer it faithfully"""

@cl.on_chat_start
def main():
  prompt = PromptTemplate(template=template,input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt,llm=llm_init,verbose=True)
  #store chain in user session
  cl.user_session.set("llm_chain",llm_chain)

@cl.on_message
async def main(message: str):
  #retrieve chain from user session
  llm_chain = cl.user_session.get('llm_chain')
  #call chain asynchronously
  res = await llm_chain.acall(message,callbacks = [cl.AsyncLangchainCallbackHandler()])
  #return the result
  await cl.Message(content=res['text']).send()

# query = """Tell me something parts of brain
#          associations and activations that happen during drea"""

# result = llm_init(query)
# print(result)


@cl.on_chat_start
def main():
  prompt = PromptTemplate(template=template,input_variables =["question"])
  llm_chain  = LLMChain(prompt=prompt,llm=llm_init)
  cl.user_session.set("llm_chain",llm_chain)

@cl.on_message
async def main(message : str):
  llm_chain = cl.user_session.get('llm_chain')
  res = await llm_chain.acall(message,callbacks=[cl.AsyncLangchainCallbackHandler()])
  await cl.Message(content=res['text']).send()