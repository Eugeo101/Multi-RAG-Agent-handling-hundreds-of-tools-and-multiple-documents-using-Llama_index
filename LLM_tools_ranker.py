import asyncio
import json
from typing import List
from llama_index.core.tools import BaseTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import AgentWorkflow
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
import os

async def score_tool(i, tool, query, llm):
    prompt = f"""Score this tool for the query (0-10):
Query: {query}
Tool: {tool.metadata.name} — {tool.metadata.description}
Respond ONLY with JSON: {{"now_score": 8, "later_score": 3, "reason": "..."}}"""
    
    response = await llm.acomplete(prompt)
    result = json.loads(response.text.strip().replace("```json","").replace("```",""))
    result["tool_index"] = i
    return result

async def llm_rerank_tools(query, tools, llm, top_k=3):
    scores = []
    for i, t in enumerate(tools):
        score = await score_tool(i, t, query, llm)
        scores.append(score)

    scores.sort(key = lambda x: x["now_score"] * 0.7 + x["later_score"] * 0.3, reverse=True)
    
    print("🔍 Tool ranking:")
    for s in scores[:top_k]:
        print(f"[{s['now_score']},{s['later_score']}] {tools[s['tool_index']].metadata.name} — {s['reason']}")
    
    return [tools[s["tool_index"]] for s in scores[:top_k]], scores

async def smart_tool_retriever(query: str, tools, llm, top_k=3):
    # can have obj_retriever layer before it (but it fails so won't add it)
    ranked_tools, scores = await llm_rerank_tools(query=query, tools=tools, llm=llm, top_k=top_k)
    return ranked_tools, scores

def load_memory(path="./memory.json") -> ChatMemoryBuffer:
    from llama_index.core.llms import ChatMessage, MessageRole
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    if os.path.exists(path):
        messages = json.load(open(path))
        for m in messages:
            memory.put(ChatMessage(
                role=MessageRole(m["role"]),
                content=m["content"]
            ))
    return memory

async def main():
    # run it
    load_dotenv('.env', override=True)
    all_tools = []
    memory = load_memory(path='./chat_1.json')
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)
    llm = GoogleGenAI(
        model="models/gemini-3.1-flash-lite-preview",
        api_key=os.getenv("google_api_key")
    )

    query = "Compare MetaGPT and LongLora approaches to training efficiency"
    best_tools, scores = await smart_tool_retriever(query, obj_retriever, llm, top_k=3)

    # use selected tools for this query only
    agent_workflow = AgentWorkflow.from_tools_or_functions(
        best_tools,
        llm=llm,
        system_prompt="Always use tools. Do not rely on prior knowledge."
    )
    response = await agent_workflow.run(user_msg=query, memory=memory)
    print(str(response))

if __name__ == '__main__':
    main()