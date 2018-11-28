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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "patch", "workspace_and_buildfile")

def _clone_or_update(ctx):
    if ((not ctx.attr.tag and not ctx.attr.commit and not ctx.attr.branch) or
        (ctx.attr.tag and ctx.attr.commit) or
        (ctx.attr.tag and ctx.attr.branch) or
        (ctx.attr.commit and ctx.attr.branch)):
        fail("Exactly one of commit, tag, or branch must be provided")
    shallow = ""
    if ctx.attr.commit:
        ref = ctx.attr.commit
    elif ctx.attr.tag:
        ref = "tags/" + ctx.attr.tag
        shallow = "--depth=1"
    else:
        ref = ctx.attr.branch
        shallow = "--depth=1"
    directory = str(ctx.path("."))
    if ctx.attr.strip_prefix:
        directory = directory + "-tmp"
    if ctx.attr.shallow_since:
        if ctx.attr.tag:
            fail("shallow_since not allowed if a tag is specified; --depth=1 will be used for tags")
        if ctx.attr.branch:
            fail("shallow_since not allowed if a branch is specified; --depth=1 will be used for branches")
        shallow = "--shallow-since=%s" % ctx.attr.shallow_since

    ctx.report_progress("Cloning %s of %s" % (ref, ctx.attr.remote))
    if (ctx.attr.verbose):
        print("git.bzl: Cloning or updating %s repository %s using strip_prefix of [%s]" %
              (
                  " (%s)" % shallow if shallow else "",
                  ctx.name,
                  ctx.attr.strip_prefix if ctx.attr.strip_prefix else "None",
              ))
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    st = ctx.execute([bash_exe, "-c", """
set -ex
( cd {working_dir} &&
    if ! ( cd '{dir_link}' && [[ "$(git rev-parse --git-dir)" == '.git' ]] ) >/dev/null 2>&1; then
      rm -rf '{directory}' '{dir_link}'
      git clone {shallow} '{remote}' '{directory}' || git clone '{remote}' '{directory}'
    fi
    git -C '{directory}' reset --hard {ref} || \
    ((git -C '{directory}' fetch {shallow} origin {ref}:{ref} || \
      git -C '{directory}' fetch origin {ref}:{ref}) && git -C '{directory}' reset --hard {ref})
      git -C '{directory}' clean -xdf )
  """.format(
        working_dir = ctx.path(".").dirname,
        dir_link = ctx.path("."),
        directory = directory,
        remote = ctx.attr.remote,
        ref = ref,
        shallow = shallow,
    )], environment = ctx.os.environ)

    if st.return_code:
        fail("error cloning %s:\n%s" % (ctx.name, st.stderr))

    if ctx.attr.strip_prefix:
        dest_link = "{}/{}".format(directory, ctx.attr.strip_prefix)
        if not ctx.path(dest_link).exists:
            fail("strip_prefix at {} does not exist in repo".format(ctx.attr.strip_prefix))

        ctx.symlink(dest_link, ctx.path("."))
    if ctx.attr.init_submodules:
        ctx.report_progress("Updating submodules")
        st = ctx.execute([bash_exe, "-c", """
set -ex
(   git -C '{directory}' submodule update --init --checkout --force )
  """.format(
            directory = ctx.path("."),
        )], environment = ctx.os.environ)
    if st.return_code:
        fail("error updating submodules %s:\n%s" % (ctx.name, st.stderr))

    ctx.report_progress("Recording actual commit")

    # After the fact, determine the actual commit and its date
    actual_commit = ctx.execute([
        bash_exe,
        "-c",
        "(git -C '{directory}' log -n 1 --pretty='format:%H')".format(
            directory = ctx.path("."),
        ),
    ]).stdout
    shallow_date = ctx.execute([
        bash_exe,
        "-c",
        "(git -C '{directory}' log -n 1 --pretty='format:%cd' --date=raw)".format(
            directory = ctx.path("."),
        ),
    ]).stdout
    return {"commit": actual_commit, "shallow_since": shallow_date}

def _remove_dot_git(ctx):
    # Remove the .git directory, if present
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    ctx.execute([
        bash_exe,
        "-c",
        "rm -rf '{directory}'".format(directory = ctx.path(".git")),
    ])

def _update_commit(orig, keys, override):
    # Merge the override information into the dict, resulting by taking the
    # given keys, as well as the name, from orig (if present there).
    result = {}
    for key in keys:
        if getattr(orig, key) != None:
            result[key] = getattr(orig, key)
    result["name"] = orig.name
    result.update(override)

    # if we found the actual commit, remove all other means of specifying it,
    # like tag or branch.
    if "commit" in result:
        result.pop("tag", None)
        result.pop("branch", None)
    return result

_common_attrs = {
    "remote": attr.string(mandatory = True),
    "commit": attr.string(default = ""),
    "shallow_since": attr.string(default = ""),
    "tag": attr.string(default = ""),
    "branch": attr.string(default = ""),
    "init_submodules": attr.bool(default = False),
    "verbose": attr.bool(default = False),
    "strip_prefix": attr.string(default = ""),
    "patches": attr.label_list(default = []),
    "patch_tool": attr.string(default = "patch"),
    "patch_args": attr.string_list(default = ["-p0"]),
    "patch_cmds": attr.string_list(default = []),
}

_new_git_repository_attrs = dict(_common_attrs.items() + {
    "build_file": attr.label(allow_single_file = True),
    "build_file_content": attr.string(),
    "workspace_file": attr.label(),
    "workspace_file_content": attr.string(),
}.items())

def _new_git_repository_implementation(ctx):
    if ((not ctx.attr.build_file and not ctx.attr.build_file_content) or
        (ctx.attr.build_file and ctx.attr.build_file_content)):
        fail("Exactly one of build_file and build_file_content must be provided.")
    update = _clone_or_update(ctx)
    workspace_and_buildfile(ctx)
    patch(ctx)
    _remove_dot_git(ctx)
    return _update_commit(ctx.attr, _new_git_repository_attrs.keys(), update)

def _git_repository_implementation(ctx):
    update = _clone_or_update(ctx)
    patch(ctx)
    _remove_dot_git(ctx)
    return _update_commit(ctx.attr, _common_attrs.keys(), update)

new_git_repository = repository_rule(
    implementation = _new_git_repository_implementation,
    attrs = _new_git_repository_attrs,
)
"""Clone an external git repository.

Clones a Git repository, checks out the specified tag, or commit, and
makes its targets available for binding. Also determine the id of the
commit actually checked out and its date, and return a dict with parameters
that provide a reproducible version of this rule (which a tag not necessarily
is).

Args:
  name: A unique name for this repository.

  build_file: The file to use as the BUILD file for this repository.
    Either build_file or build_file_content must be specified.

    This attribute is an absolute label (use '@//' for the main repo). The file
    does not need to be named BUILD, but can be (something like
    BUILD.new-repo-name may work well for distinguishing it from the
    repository's actual BUILD files.

  build_file_content: The content for the BUILD file for this repository.
    Either build_file or build_file_content must be specified.

  workspace_file: The file to use as the `WORKSPACE` file for this repository.

    Either `workspace_file` or `workspace_file_content` can be specified, or
    neither, but not both.
  workspace_file_content: The content for the WORKSPACE file for this repository.

    Either `workspace_file` or `workspace_file_content` can be specified, or
    neither, but not both.
  branch: branch in the remote repository to checked out

  tag: tag in the remote repository to checked out

  commit: specific commit to be checked out
    Precisely one of branch, tag, or commit must be specified.

  shallow_since: an optional date, not after the specified commit; the
    argument is not allowed if a tag is specified (which allows cloning
    with depth 1). Setting such a date close to the specified commit
    allows for a more shallow clone of the repository, saving bandwidth and
    wall-clock time.

  init_submodules: Whether to clone submodules in the repository.

  remote: The URI of the remote Git repository.

  strip_prefix: A directory prefix to strip from the extracted files.

  patches: A list of files that are to be applied as patches after extracting
    the archive.
  patch_tool: the patch(1) utility to use.
  patch_args: arguments given to the patch tool, defaults to ["-p0"]
  patch_cmds: sequence of commands to be applied after patches are applied.
"""

git_repository = repository_rule(
    implementation = _git_repository_implementation,
    attrs = _common_attrs,
)
"""Clone an external git repository.

Clones a Git repository, checks out the specified tag, or commit, and
makes its targets available for binding. Also determine the id of the
commit actually checked out and its date, and return a dict with parameters
that provide a reproducible version of this rule (which a tag not necessarily
is).


Args:
  name: A unique name for this repository.

  init_submodules: Whether to clone submodules in the repository.

  remote: The URI of the remote Git repository.

  branch: branch in the remote repository to checked out

  tag: tag in the remote repository to checked out

  commit: specific commit to be checked out
    Precisely one of branch, tag, or commit must be specified.

  shallow_since: an optional date in the form YYYY-MM-DD, not after
    the specified commit; the argument is not allowed if a tag is specified
    (which allows cloning with depth 1). Setting such a date close to the
    specified commit allows for a more shallow clone of the repository, saving
    bandwidth and wall-clock time.

  strip_prefix: A directory prefix to strip from the extracted files.

  patches: A list of files that are to be applied as patches after extracting
    the archive.
  patch_tool: the patch(1) utility to use.
  patch_args: arguments given to the patch tool, defaults to ["-p0"]
  patch_cmds: sequence of commands to be applied after patches are applied.
"""
