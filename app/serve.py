from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/nlp-foodie/playground")

from chain import chain as nlp_foodie_chain
add_routes(app, nlp_foodie_chain, path="/nlp-foodie")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
