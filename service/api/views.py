import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger
from service.models.popular import get_popular_items
from service.models import ann
from service.utils import load_model


RECOMMENDATIONS = Dict[str, List[int]]

models: Dict[str, Any] = {}
user_knn_offline: RECOMMENDATIONS = {}
lightfm_offline: RECOMMENDATIONS = {}
popular: List[int] = []


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
    global models, user_knn_offline, lightfm_offline, popular

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
    if model_name == 'user_knn':
        items = user_knn_offline.get(str(user_id))
        if items is None:
            try:
                predict = models[model_name].predict_user(
                    user_id=user_id,
                    n_recs=20
                )[:k_recs]
                items = predict.item_id.tolist()[:k_recs]
            except KeyError:
                items = []
        else:
            items = items[:k_recs]
    elif model_name == 'lightfm_warp_12':
        items = lightfm_offline.get(str(user_id), [])
        items = items[:k_recs]  # type: ignore
    elif model_name == 'ann_over_lightfm':
        try:
            items = models[model_name].predict(user_id=user_id).tolist()[:k_recs]
        except (KeyError, IndexError):
            items = []

    recs = [p for p in popular if p not in items]  # type: ignore
    recs = recs[:k_recs - len(items)]  # type: ignore
    items = list(items) + list(recs)  # type: ignore
    return RecoResponse(user_id=user_id, items=items)


def add_views(app: FastAPI) -> None:
    app.include_router(router)

    @app.on_event("startup")
    async def startup_event():
        global models, user_knn_offline, lightfm_offline, popular
        models['user_knn'] = load_model(model_path='models/userknn.pickle')
        models['lightfm_warp_12'] = True
        models['ann_over_lightfm'] = ann.get_ann(k_reco=app.state.k_recs)
        with open('offline/lightfm_warp_12.json') as off:
            lightfm_offline.update(json.load(off))
        with open('offline/user_knn.json') as off:
            user_knn_offline.update(json.load(off))

        items = get_popular_items('default', app.state.k_recs)
        popular += items
