# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Respositry rules for downloading apt-get repositories."""

def _run_cmd(ctx, cmd):
  """Helper for running commands."""
  if ctx.which(cmd[0]) == None:
    fail("%s not found" % cmd[0])
  result = ctx.execute(cmd)
  if result.return_code != 0:
    print("stdout: %s, stderr: %s" % (result.stdout, result.stderr))
    fail("Running '%s' failed" % " ".join(cmd))
  return result

def _impl(ctx):
  """Download .deb and extract."""
  result = _run_cmd(ctx, ["apt-get", "download", "--print-uris", ctx.name])
  # --print-uris output is in the format: URL filename SHA256:sha
  deb = result.stdout.split(" ")[1]
  _run_cmd(ctx, ["apt-get", "download", ctx.name])
  _run_cmd(ctx, ["ar", "x", deb])
  _run_cmd(ctx, ["tar", "xf", "data.tar.xz"])

  ctx.file("WORKSPACE", "workspace(name = '%s')" % ctx.name)
  ctx.file("BUILD", "exports_files(%s)" % ctx.attr.exports_files)

apt_get = repository_rule(
    implementation = _impl,
    attrs = {"exports_files": attr.string_list(default=[])},
)
