from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from Phase2.Agents.orchestrator_agent import OrchestratorAgent

app = FastAPI()

templates = Jinja2Templates(directory="ui/templates")

orchestrator = OrchestratorAgent()


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "response": ""}
    )


@app.post("/ask")
async def ask(request: Request):

    form = await request.form()
    user_query = form.get("query")

    result = orchestrator.handle_request(user_query)

    import json

    if result.data:
        response = result.message + "\n" + json.dumps(result.data, indent=2)
    else:
        response = result.message

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "response": response
        }
    )
