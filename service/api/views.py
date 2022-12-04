from typing import List

import pandas as pd
from fastapi import APIRouter, FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from service.api.exceptions import UserNotFoundError, ModelNotFoundError
from service.log import app_logger
from service.utils import load_model


models = {}


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()
auth_scheme = HTTPBearer(auto_error=False)


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        200: {"description": "Return recommendations for users."},
        401: {"description": "You are not authenticated"},
        404: {"description": "The Model or User was not found"},
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme)
) -> RecoResponse:
    global models

    if not token or token.credentials != request.app.state.api_key:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name not in models:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    k_recs = request.app.state.k_recs
    pred = models[model_name].predict(test=pd.DataFrame({'user_id': [user_id]}))[:k_recs]
    return RecoResponse(user_id=user_id, items=list(pred.item_id))


def add_views(app: FastAPI) -> None:
    app.include_router(router)
    @app.on_event("startup")
    async def startup_event():
        global models
        models['user_knn'] = load_model(model_path='models/userknn.pickle')
