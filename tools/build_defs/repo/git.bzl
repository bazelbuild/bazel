# Copyright 2015 The Bazel Authors. All rights reserved.
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
"""Rules for cloning external git repositories."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "workspace_and_buildfile", "patch")


def _clone_or_update(ctx):
  if ((not ctx.attr.tag and not ctx.attr.commit) or
      (ctx.attr.tag and ctx.attr.commit)):
    fail('Exactly one of commit and tag must be provided')
  shallow = ''
  if ctx.attr.commit:
    ref = ctx.attr.commit
  else:
    ref = 'tags/' + ctx.attr.tag
    shallow = '--depth=1'
  directory=str(ctx.path('.'))
  if ctx.attr.strip_prefix:
    directory = directory + "-tmp"
  if ctx.attr.shallow_since:
    if ctx.attr.tag:
      fail('shallow_since not allowed if a tag is specified; --depth=1 will be used for tags')
    shallow='--shallow-since=%s' % ctx.attr.shallow_since

  if (ctx.attr.verbose):
    print('git.bzl: Cloning or updating%s repository %s using strip_prefix of [%s]' %
    (' (%s)' % shallow if shallow else '',
     ctx.name,
     ctx.attr.strip_prefix if ctx.attr.strip_prefix else 'None',
    ))
  bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
  st = ctx.execute([bash_exe, '-c', """
set -ex
( cd {working_dir} &&
    if ! ( cd '{dir_link}' && [[ "$(git rev-parse --git-dir)" == '.git' ]] ) >/dev/null 2>&1; then
      rm -rf '{directory}' '{dir_link}'
      git clone {shallow} '{remote}' '{directory}'
    fi
    cd '{directory}'
    git reset --hard {ref} || (git fetch {shallow} origin {ref}:{ref} && git reset --hard {ref})
    git clean -xdf )
  """.format(
      working_dir=ctx.path('.').dirname,
      dir_link=ctx.path('.'),
      directory=directory,
      remote=ctx.attr.remote,
      ref=ref,
      shallow=shallow,
  )])

  if st.return_code:
    fail('error cloning %s:\n%s' % (ctx.name, st.stderr))

  if ctx.attr.strip_prefix:
    dest_link="{}/{}".format(directory, ctx.attr.strip_prefix)
    if not ctx.path(dest_link).exists:
      fail("strip_prefix at {} does not exist in repo".format(ctx.attr.strip_prefix))

    ctx.symlink(dest_link, ctx.path('.'))
  if ctx.attr.init_submodules:
    st = ctx.execute([bash_exe, '-c', """
set -ex
(   cd '{directory}'
    git submodule update --init --checkout --force )
  """.format(
      directory=ctx.path('.'),
  )])
  if st.return_code:
    fail('error updating submodules %s:\n%s' % (ctx.name, st.stderr))


def _new_git_repository_implementation(ctx):
  if ((not ctx.attr.build_file and not ctx.attr.build_file_content) or
      (ctx.attr.build_file and ctx.attr.build_file_content)):
    fail('Exactly one of build_file and build_file_content must be provided.')
  _clone_or_update(ctx)
  workspace_and_buildfile(ctx)
  patch(ctx)

def _git_repository_implementation(ctx):
  _clone_or_update(ctx)
  patch(ctx)


_common_attrs = {
    'remote': attr.string(mandatory=True),
    'commit': attr.string(default=''),
    'shallow_since': attr.string(default=''),
    'tag': attr.string(default=''),
    'init_submodules': attr.bool(default=False),
    'verbose': attr.bool(default=False),
    'strip_prefix': attr.string(default=''),
    'patches': attr.label_list(default=[]),
    'patch_tool': attr.string(default="patch"),
    'patch_cmds': attr.string_list(default=[]),
}


new_git_repository = repository_rule(
    implementation = _new_git_repository_implementation,
    attrs = dict(_common_attrs.items() + {
        'build_file': attr.label(allow_single_file=True),
        'build_file_content': attr.string(),
    }.items())
)
"""Clone an external git repository.

Clones a Git repository, checks out the specified tag, or commit, and
makes its targets available for binding.

Args:
  name: A unique name for this rule.

  build_file: The file to use as the BUILD file for this repository.
    Either build_file or build_file_content must be specified.

    This attribute is a label relative to the main workspace. The file
    does not need to be named BUILD, but can be (something like
    BUILD.new-repo-name may work well for distinguishing it from the
    repository's actual BUILD files.

  build_file_content: The content for the BUILD file for this repository.
    Either build_file or build_file_content must be specified.

  tag: tag in the remote repository to checked out

  commit: specific commit to be checked out
    Either tag or commit must be specified.

  shallow_since: an optional date, not after the specified commit; the
    argument is not allowed if a tag is specified (which allows cloning
    with depth 1). Setting such a date close to the specified commit
    allows for a more shallow clone of the repository, saving bandwith and
    wall-clock time.

  init_submodules: Whether to clone submodules in the repository.

  remote: The URI of the remote Git repository.

  strip_prefix: A directory prefix to strip from the extracted files.

  patches: A list of files that are to be applied as patches after extracting
    the archive.
  patch_tool: the patch(1) utility to use.
  patch_cmds: sequence of commands to be applied after patches are applied.
"""

git_repository = repository_rule(
    implementation=_git_repository_implementation,
    attrs=_common_attrs,
)
"""Clone an external git repository.

Clones a Git repository, checks out the specified tag, or commit, and
makes its targets available for binding.

Args:
  name: A unique name for this rule.

  init_submodules: Whether to clone submodules in the repository.

  remote: The URI of the remote Git repository.

  tag: tag in the remote repository to checked out

  commit: specific commit to be checked out
    Either tag or commit must be specified.

  shallow_since: an optional date, not after the specified commit; the
    argument is not allowed if a tag is specified (which allows cloning
    with depth 1). Setting such a date close to the specified commit
    allows for a more shallow clone of the repository, saving bandwith and
    wall-clock time.

  strip_prefix: A directory prefix to strip from the extracted files.

  patches: A list of files that are to be applied as patches after extracting
    the archive.
  patch_tool: the patch(1) utility to use.
  patch_cmds: sequence of commands to be applied after patches are applied.
"""
