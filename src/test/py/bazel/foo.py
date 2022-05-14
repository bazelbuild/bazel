# standard libraries
import os
import subprocess
import sys

# third party libraries
from rules_python.python.runfiles import runfiles

r = runfiles.Create()
env = dict(os.environ)
env.update(r.EnvVars())
b = r.Rlocation("io_bazel/src/test/py/bazel/bar")
print(b)
if not b:
    print("Could not find bar binary.")
    sys.exit(1)
subprocess.run([b], env=env)
print("goodbye world.")
