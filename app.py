import contextlib
import json
import os
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from pydantic import BaseModel

load_dotenv()

from analysis_agent.agent import root_agent, hypothesis_agent  # noqa: E402

APP_NAME = "movie_analysis_app"

# Separate runners for each agent
session_service = InMemorySessionService()

root_runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

hyp_session_service = InMemorySessionService()
hyp_runner = Runner(
    agent=hypothesis_agent,
    app_name=APP_NAME + "_hyp",
    session_service=hyp_session_service,
)

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    chart_data: dict | None = None


@app.get("/")
async def index():
    return FileResponse("index.html")


async def _run_agent(runner, svc, app_name, user_id, session_id, message):
    """Run an agent and return (response_text, func_responses_dict)."""
    with contextlib.suppress(Exception):
        await svc.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id,
        )

    content = Content(role="user", parts=[Part.from_text(text=message)])
    response_text = ""
    func_results = {}

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content,
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if part.text:
                response_text = part.text
            fr = getattr(part, "function_response", None)
            if fr:
                name = getattr(fr, "name", "")
                resp = getattr(fr, "response", None)
                if isinstance(resp, dict) and resp.get("result"):
                    func_results[name] = resp["result"]

    return response_text, func_results


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    # --- Step 1+2: Root agent collects and analyzes ---
    root_text, root_funcs = await _run_agent(
        root_runner, session_service, APP_NAME,
        session_id, session_id, request.message,
    )

    stats_json = root_funcs.get("_compute_stats", "")
    anomalies_json = root_funcs.get("_detect_anomalies", "")

    # --- Step 3: Hypothesis agent writes narrative ---
    response_text = root_text  # fallback to root agent's text
    if stats_json and anomalies_json:
        hyp_session_id = str(uuid.uuid4())
        hyp_message = (
            f"User question: {request.message}\n\n"
            f"summary_stats: {stats_json}\n\n"
            f"anomalies: {anomalies_json}"
        )
        hyp_text, _ = await _run_agent(
            hyp_runner, hyp_session_service, APP_NAME + "_hyp",
            hyp_session_id, hyp_session_id, hyp_message,
        )
        if hyp_text:
            response_text = hyp_text

    # --- Generate charts server-side ---
    chart_data = None
    if stats_json:
        try:
            from analysis_agent.tools.hypothesis_tools import _build_chart_data
            chart_result = _build_chart_data(
                json.dumps({"summary_stats": json.loads(stats_json)})
            )
            chart_data = json.loads(chart_result).get("chart_data")
        except Exception:
            pass

    # --- Save artifact server-side ---
    if response_text and stats_json and len(response_text) > 100:
        try:
            from analysis_agent.tools.artifact_tools import _save_report
            _save_report(
                question=request.message,
                hypothesis=response_text[:500],
                evidence_bullets="(see full response)",
                movie_count=json.loads(stats_json).get("total_movies", 0),
                dataset_label="Movie Analysis",
            )
        except Exception:
            pass

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        chart_data=chart_data,
    )


@app.post("/clear")
async def clear(session_id: str | None = None):
    if session_id:
        with contextlib.suppress(Exception):
            await session_service.delete_session(
                app_name=APP_NAME, user_id=session_id, session_id=session_id,
            )
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
