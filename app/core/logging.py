from __future__ import annotations

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"level":record.levelname,"logger":record.name,"message":record.getMessage(),"timestamp":self.formatTime(record),"request_id":getattr(record,"request_id",None)})


def configure_logging() -> None:
    handler=logging.StreamHandler(sys.stdout); handler.setFormatter(JsonFormatter()); root=logging.getLogger(); root.handlers[:]=[handler]; root.setLevel(logging.INFO)
