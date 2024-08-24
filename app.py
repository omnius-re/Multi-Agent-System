from typing_extensions import TypedDict
from typing import AsyncIterable, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from langchain.callbacks import AsyncIteratorCallbackHandler
from langserve import add_routes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import asyncio
import uvicorn

from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph, START


# Environment variablespi
api_key = os.getenv("OPENAI_API_KEY")

# FastAPI application setup
app = FastAPI(
    title = "Agent Server",
    version = 1.0,
    description = "A simple API server"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the prompt template
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in Python programming. 
            Answer the user question. Provide a description of the code solution, 
            then list the imports, and finally the functioning code block. 
            Here is the user question:""",
        ),
        ("user", "{messages}"),
    ]
)

# Data model for the request
class CodeQuery(BaseModel):
    messages: str

# Data model for the response
class CodeResponse(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

# STATE:
class GraphState(TypedDict):
    messages: List[str]

# Define the chain globally
callback = AsyncIteratorCallbackHandler()
llm = ChatOpenAI(
    streaming=True, 
    verbose=True,
    callbacks=[callback],
    temperature=0, 
    api_key=api_key,
    model="gpt-4o-mini", 
)

add_routes(
    app,
    llm,
    path="/openai"
)

code_gen_chain = code_gen_prompt | llm.with_structured_output(CodeResponse)

add_routes(
    app,
    code_gen_chain,
    path="/chain"
)

async def generate(query: str) -> AsyncIterable[str]:
    task = asyncio.create_task(
        llm.agenerate({"messages": query})
    )

    try:
        async for token in callback.aiter():
            yield f"data: {token}\n\n"
    except Exception as e:
        yield f"data: Error: {e}\n\n"
    finally:
        callback.done.set()

    await task  

@app.post("/stream_chat/")
async def stream_chat(query: CodeQuery):
    generator = generate(query.messages)
    return StreamingResponse(generator, media_type="text/event-stream")


# Nodes
def generate(state: GraphState):
    messages = state["messages"]
    code_solution = code_gen_chain.invoke({"messages": [("user", messages[-1])]})
    messages.append(f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}")
    return {"messages": messages}

def code_check(state: GraphState):
    messages = state["messages"]
    code_solution = state["messages"][-1]
    
    imports = code_solution.imports
    code = code_solution.code

    try:
        exec(imports)
    except Exception as e:
        messages.append(f"Your solution failed the import test: {e}")
        return {"messages": messages}

    try:
        exec(imports + "\n" + code)
    except Exception as e:
        messages.append(f"Your solution failed the code execution test: {e}")
        return {"messages": messages}

    return {"messages": messages}

def reflect(state: GraphState):
    messages = state["messages"]
    reflections = code_gen_chain.invoke({"messages": [("user", messages[-1])]})
    messages.append(f"Here are reflections on the error: {reflections}")
    return {"messages": messages}

def decide_to_finish(state: GraphState):
    messages = state["messages"]
    # Determine when to end or continue
    return "end"

# Workflow setup
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)
workflow.add_node("check_code", code_check)
workflow.add_node("reflect", reflect)

# Build graph
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "reflect": "reflect",
    },
)
workflow.add_edge("reflect", "generate")

compiled_app = workflow.compile()

def process_state(state: GraphState) -> GraphState:
    return compiled_app.invoke(state)

add_routes(
    app,
    compiled_app,
    path="/agents"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)