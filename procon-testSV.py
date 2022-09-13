import os
import sys
import urllib.parse
import html
import json

from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from http import HTTPStatus

PORT = 80
TOKEN = 'NULL'
result = 1
#-------------------------------------------------------
data={"procon-token":TOKEN}
#-------------------------------------------------------

class StubHttpRequestHandler(BaseHTTPRequestHandler):
    server_version = "HTTP Stub/0.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def match_GET(self):
        if result == 1:
            #試合情報の定義
            PROBLEMS = 3
            BONUS_FACTOR = [1.3,1.2,1.1,1]
            PENALTY = 3
        
        res = {"problems":PROBLEMS,"bonus_factor":BONUS_FACTOR,"penalty":PENALTY}


        self.send_response(json.dumps(res))

    def problem_GET(self):
        if result == 1:
            #問題情報の定義
            ID = 'qual-1-1'
            CHUNKS = 3
            STARTS_AT = 1655302266
            TIME_LIMIT = 1000
            DATA = 3
        
        res = {"id":ID,"chunks":CHUNKS,"starts_at":STARTS_AT,"time_limit":TIME_LIMIT,"data":DATA}
        self.send_response(json.dump(res))


    def POST_problems(self):
        print("")

handler = StubHttpRequestHandler
httpd = HTTPServer(('',PORT),handler)
httpd.serve_forever()