# import native Python packages
import multiprocessing

# import third party packages
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

# import custom local stuff
from instance.config import GCP_FILE
from src.tarpeydev import (
    index,
    autobracket,
    ddr,
    haveyouseenx,
    mildredleague,
    timecapsule,
    users,
)
from src.api.index import index_api
from src.api.autobracket import ab_api
from src.api.haveyouseenx import hysx_api
from src.api.mildredleague import ml_api
from src.api.users import users_api
from src.db.startup import motor_startup, motor_shutdown

# GCP debugger
try:
    import googleclouddebugger
    googleclouddebugger.enable(
        breakpoint_enable_canary=False,
        service_account_json_file=GCP_FILE,
    )
except ImportError:
    pass


def multiproc_context():
    # possible multiprocessing solution from https://github.com/tiangolo/fastapi/issues/1487
    # this prevents gunicorn workers from crashing on multiprocessing!
    multiprocessing.set_start_method('spawn')


def create_fastapi_app():
    # create the root and API FastAPI apps
    view_app = FastAPI(
        title="tarpey.dev",
        description="Mike Tarpey's app playground.",
        docs_url=None,
        redoc_url=None,
        default_response_class=ORJSONResponse,
    )

    # templates
    templates = Jinja2Templates(directory='templates')

    # startup and shutdown connection to DB
    # see https://motor.readthedocs.io/en/stable/tutorial-asyncio.html
    view_app.add_event_handler('startup', multiproc_context)
    view_app.add_event_handler('startup', motor_startup)
    view_app.add_event_handler('shutdown', motor_shutdown)

    # custom exception page to convert the 422 into a 404.
    @view_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc):
        return templates.TemplateResponse(
            'index/error.html',
            context={
                'request': request,
                'exc': 'This page doesn\'t exist.',
                'status_code': 404,
            },
            status_code=404,
        )

    # custom exception page to convert a starlette (not FastAPI) exception
    @view_app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc):
        return templates.TemplateResponse(
            'index/error.html',
            context={
                'request': request,
                'exc': str(exc.detail),
                'status_code': exc.status_code,
            }
        )

    api_app = FastAPI(
        title="tarpey.dev API",
        description="API for Mike Tarpey's app sandbox.",
        servers=[
            {"url": "http://127.0.0.1:8000/api", "description": "Testing environment."},
            {"url": "https://dev.tarpey.dev/api", "description": "Staging environment."},
            {"url": "https://tarpey.dev/api", "description": "Production environment"},
        ],
        default_response_class=ORJSONResponse,
    )

    # config stuff
    view_app.mount("/static", app=StaticFiles(directory='static'), name="static")

    # include subrouters of views
    view_app.include_router(index.index_views)
    view_app.include_router(autobracket.autobracket_views)
    view_app.include_router(ddr.ddr_views)
    view_app.include_router(haveyouseenx.hysx_views)
    view_app.include_router(mildredleague.ml_views)
    view_app.include_router(timecapsule.tc_views)
    view_app.include_router(timecapsule.tcd_views)
    view_app.include_router(users.user_views)

    # include subrouters of the FastAPI app
    api_app.include_router(index_api)
    api_app.include_router(ab_api)
    api_app.include_router(hysx_api)
    api_app.include_router(ml_api)
    api_app.include_router(users_api)

    # stitch it all together
    view_app.mount("/api", app=api_app, name="api_app")

    return view_app


# entrypoint!
app = create_fastapi_app()
