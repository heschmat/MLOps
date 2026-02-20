import logging
import sys
import json
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Only include custom extra fields
        for key in getattr(record, "extra_fields", {}):
            log[key] = record.extra_fields[key]

        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)

        return json.dumps(log)


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]
