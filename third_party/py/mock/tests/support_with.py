from __future__ import with_statement

import sys

__all__ = ['nested', 'catch_warnings', 'examine_warnings']


try:
    from contextlib import nested
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def nested(*managers):
        exits = []
        vars = []
        exc = (None, None, None)
        try:
            for mgr in managers:
                exit = mgr.__exit__
                enter = mgr.__enter__
                vars.append(enter())
                exits.append(exit)
            yield vars
        except:
            exc = sys.exc_info()
        finally:
            while exits:
                exit = exits.pop()
                try:
                    if exit(*exc):
                        exc = (None, None, None)
                except:
                    exc = sys.exc_info()
            if exc != (None, None, None):
                raise exc[1]

# copied from Python 2.6
try:
    from warnings import catch_warnings
except ImportError:
    class catch_warnings(object):
        def __init__(self, record=False, module=None):
            self._record = record
            self._module = sys.modules['warnings']
            self._entered = False

        def __repr__(self):
            args = []
            if self._record:
                args.append("record=True")
            name = type(self).__name__
            return "%s(%s)" % (name, ", ".join(args))

        def __enter__(self):
            if self._entered:
                raise RuntimeError("Cannot enter %r twice" % self)
            self._entered = True
            self._filters = self._module.filters
            self._module.filters = self._filters[:]
            self._showwarning = self._module.showwarning
            if self._record:
                log = []
                def showwarning(*args, **kwargs):
                    log.append(WarningMessage(*args, **kwargs))
                self._module.showwarning = showwarning
                return log
            else:
                return None

        def __exit__(self, *exc_info):
            if not self._entered:
                raise RuntimeError("Cannot exit %r without entering first" % self)
            self._module.filters = self._filters
            self._module.showwarning = self._showwarning

    class WarningMessage(object):
        _WARNING_DETAILS = ("message", "category", "filename", "lineno", "file",
                            "line")
        def __init__(self, message, category, filename, lineno, file=None,
                        line=None):
            local_values = locals()
            for attr in self._WARNING_DETAILS:
                setattr(self, attr, local_values[attr])
            self._category_name = None
            if category.__name__:
                self._category_name = category.__name__


def examine_warnings(func):
    def wrapper():
        with catch_warnings(record=True) as ws:
            func(ws)
    return wrapper
