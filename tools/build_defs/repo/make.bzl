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
""""Respositry rules for converting Makefile-based projects to Bazel."""

def _impl(ctx):
  """Downloads, extracts, and runs configure && make on an archive."""
  ctx.download_and_extract(
      url = ctx.attr.urls,
      stripPrefix = ctx.attr.strip_prefix,
      sha256 = ctx.attr.sha256,
  )

  result = ctx.execute(['./configure'])
  if result.return_code != 0:
    print(result.stdout)
    print(result.stderr)
    fail('error running configure')
  # TODO(kchodorow): allow specifying make rules here.
  result = ctx.execute(['make'])
  if result.return_code != 0:
    print(result.stdout)
    print(result.stderr)
    fail('error running make')

  # Check that all expected files were created.
  missing_files = []
  for f in ctx.attr.exports:
    if not ctx.path(f).exists:
      missing_files += [f]
  if missing_files:
    fail('file(s) %s were not created' % ', '.join(missing_files))

  ctx.file('WORKSPACE', """
# Automatically generated WORKSPACE file.
workspace(name = 'name')
""".format(name = ctx.name))
  ctx.file('BUILD', """
exports_files({exports})

{build_file_content}
""".format(
    exports = ctx.attr.exports,
    build_file_content = ctx.attr.build_file_content))


make = repository_rule(
    implementation = _impl,
    attrs = {
        'urls':attr.string_list(mandatory = True),
        'exports':attr.string_list(mandatory = True),
        'strip_prefix':attr.string(default = ''),
        'sha256':attr.string(default = ''),
        'build_file_content':attr.string(default = ''),
    }
)
