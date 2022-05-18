# standard libraries
import os
import subprocess
import sys

def RulesPythonRlocation(path):
    """This code recreates the functionality of rules_python.python.runfiles.runfiles.Rlocation."""
    if not path:
        raise ValueError()
    if not isinstance(path, str):
        raise TypeError()
    if (
        path.startswith("../")
        or "/.." in path
        or path.startswith("./")
        or "/./" in path
        or path.endswith("/.")
        or "//" in path
    ):
        raise ValueError('path is not normalized: "%s"' % path)
    if path[0] == "\\":
        raise ValueError('path is absolute without a drive letter: "%s"' % path)
    if os.path.isabs(path):
        return path
    runfiles_root = os.environ.get("RUNFILES_DIR")
    return posixpath.join(runfiles_root, path)

r = runfiles.Create()
env = dict(os.environ)
env.update(r.EnvVars())
b = RulesPythonRlocation("io_bazel/src/test/py/bazel/buildzip_bar")
print(b)
if not b:
    print("Could not find buildzip_bar binary.")
    sys.exit(1)
subprocess.run([b], env=env)
print("goodbye world.")
