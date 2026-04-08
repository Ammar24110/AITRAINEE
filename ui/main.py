import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from Phase2.Agents.orchestrator_agent import OrchestratorAgent


app = FastAPI()
templates = Jinja2Templates(directory="ui/templates")


# 🔥 store agent in app state (IMPORTANT)
@app.on_event("startup")
async def startup_event():
    app.state.agent = OrchestratorAgent()
    await app.state.agent.initialize_mcp()


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "response": ""}
    )


@app.post("/ask")
async def ask(request: Request):
    form = await request.form()
    user_query = form.get("query", "").strip()

    if not user_query:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "response": "Please enter a valid query."
            }
        )

    try:
        # 🔥 get agent from app state
        agent = request.app.state.agent

        response = await agent.handle_request(user_query)

        if not response:
            response = "No response generated."

        response = str(response).replace("\n", "<br>")

    except Exception as e:
        print("FULL ERROR:", e)
        response = f"Error: {str(e)}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "response": response
        }
    )


@app.on_event("shutdown")
async def shutdown_event():
    agent = getattr(app.state, "agent", None)

    if agent and hasattr(agent, "shutdown"):
        await agent.shutdown()