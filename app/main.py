from fastapi import FastAPI, Depends
from fastapi.security import APIKeyHeader

auth_header = APIKeyHeader(name="Authorization", auto_error=False)

app = FastAPI(dependencies=[Depends(auth_header)])
