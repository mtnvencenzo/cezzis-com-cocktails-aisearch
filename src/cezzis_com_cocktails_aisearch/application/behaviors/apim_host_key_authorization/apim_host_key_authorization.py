import logging
import os
from functools import wraps
from typing import cast

from fastapi import HTTPException, Request

from cezzis_com_cocktails_aisearch.domain.config.app_options import AppOptions, get_app_options

_logger = logging.getLogger("apim_host_key_authorization")
_app_options: AppOptions = get_app_options()


def apim_host_key_authorization(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = cast(Request, kwargs.get("_rq"))
        supplied_host_key = request.headers.get("X-Apim-Host-Key", "")

        if len(_app_options.apim_host_key.strip()) == 0:
            if os.getenv("ENV") != "local":
                _logger.warning("Host key authorization bypassed due to unconfigured host key")
        else:
            if supplied_host_key != _app_options.apim_host_key:
                _logger.warning("Host key authorization failed due to invalid supplied host key")
                raise HTTPException(status_code=403, detail="Invalid host key")

        return await func(*args, **kwargs)

    return wrapper
