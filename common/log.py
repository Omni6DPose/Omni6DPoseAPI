"""
Logging
=======

In principle, we recommend Python's official `logging` package for logging.
This module contains utilities for logging setup.
"""

import logging
import json
import sys

try:
    from typing import Any, Literal
except:
    pass


class _CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    FORMAT_SHORT = " %(asctime)s [%(name)s] %(message)s"
    FORMAT_LONG = " %(asctime)s [%(name)s] %(message)s (%(filename)s:%(lineno)d)"
    FILE_FORMAT = (
        "%(levelname)s %(asctime)s [%(name)s] %(message)s (%(filename)s:%(lineno)d)"
    )

    LEVEL_FORMATS = {
        logging.DEBUG: ">> " + grey + "%(levelname)s" + reset + FORMAT_LONG,
        logging.INFO: green + "%(levelname)s" + reset + FORMAT_SHORT,
        logging.WARNING: yellow + "%(levelname)s" + reset + FORMAT_LONG,
        logging.ERROR: red + "%(levelname)s" + reset + FORMAT_LONG,
        logging.CRITICAL: bold_red + "%(levelname)s" + reset + FORMAT_LONG,
    }

    mode: str

    def __init__(self, mode: 'Literal["terminal", "file"]' = "terminal") -> None:
        self.mode = mode
        super().__init__()

    def format(self, record):
        if self.mode == "terminal":
            level_fmt = self.LEVEL_FORMATS.get(record.levelno)
            formatter = logging.Formatter(level_fmt)
        else:
            formatter = logging.Formatter(self.FILE_FORMAT)

        return formatter.format(record)


class _JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.

    :param fmt_dict: Key: logging format attribute pairs. Defaults to ``{"message": "message"}``.
    :param time_format: time.strftime() format string. Default: "%Y-%m-%dT%H:%M:%S"
    :param msec_format: Microsecond formatting. Appended at the end. Default: "%s.%03dZ"
    """

    def __init__(
        self,
        fmt_dict: dict = {
            "level": "levelname",
            "message": "message",
            "loggerName": "name",
            "processName": "processName",
            "processID": "process",
            "threadName": "threadName",
            "threadID": "thread",
            "timestamp": "asctime",
        },
        time_format: str = "%Y-%m-%dT%H:%M:%S",
        msec_format: str = "%s.%03dZ",
    ):
        self.fmt_dict = fmt_dict if fmt_dict is not None else {"message": "message"}
        self.default_time_format = time_format
        self.default_msec_format = msec_format
        self.datefmt = None

    def usesTime(self) -> bool:
        """
        Overwritten to look for the attribute in the format dict values instead of the fmt string.
        """
        return "asctime" in self.fmt_dict.values()

    def formatMessage(self, record) -> dict:
        """
        Overwritten to return a dictionary of the relevant LogRecord attributes instead of a string.
        KeyError is raised if an unknown attribute is provided in the fmt_dict.
        """
        return {
            fmt_key: record.__dict__[fmt_val]
            for fmt_key, fmt_val in self.fmt_dict.items()
        }

    def format(self, record) -> str:
        """
        Mostly the same as the parent's class method, the difference being that a dict is manipulated and dumped as JSON
        instead of a string.
        """
        record.message = record.getMessage()

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message_dict = self.formatMessage(record)

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(message_dict, default=str)


def setup_logging(
    log_mode: str, debug=False, logfile: "str | None" = None, json=True, delay=True
):
    """Setup a simple logging for Python's official ``logging`` package.

    :param log_mode: ``file`` for only logging to file, `terminal` for only logging to terminal, otherwise both are taken.
    :param debug: whether set loglevel to DEBUG (default is INFO)
    :param logfile: specify logging filename
    :param json: whether output json format log, whose file name is ``logfile + ".jl"``.
    :param delay: See `FileHandler <https://docs.python.org/3/library/logging.handlers.html#filehandler>`_

    .. highlight:: python

    Usage::

        # only logging to terminal, enabling debug level logging.
        setup_logging("terminal", debug=True)

        # only logging to runtime.log file and runtime.log.jl (json log)
        setup_logging("file", logfile="runtime.log")

        # logging to terminal and file without json
        setup_logging("both", logfile="runtime.log", json=false)

    """
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[0])

    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.root.setLevel(level=level)

    if log_mode != "file":
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(_CustomFormatter("terminal"))
        logging.root.addHandler(handler)

    if log_mode != "terminal":
        assert logfile is not None
        handler = logging.FileHandler(logfile, delay=delay)
        handler.setLevel(level)
        handler.setFormatter(_CustomFormatter("file"))
        logging.root.addHandler(handler)
        if json:
            handler = logging.FileHandler(logfile + ".jl", delay=delay)
            handler.setLevel(level)
            handler.setFormatter(_JsonFormatter())
            logging.root.addHandler(handler)
