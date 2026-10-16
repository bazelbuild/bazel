"""Example for using runfiles with Python."""

import pathlib

from python.runfiles import runfiles

SOME_FILE = pathlib.Path("examples/runfile.txt")

r = runfiles.Create()
root = r.CurrentRepository()
if not root:
  root = "_main"
realPathToSomeFile = r.Rlocation(str(root / SOME_FILE))

print("The content of the runfile is:")
with open(realPathToSomeFile, "r") as f:
  print(f.read(), end="")
