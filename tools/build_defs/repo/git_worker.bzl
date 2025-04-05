# Copyright 2019 The Bazel Authors. All rights reserved.
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

# WARNING:
# https://github.com/bazelbuild/bazel/issues/17713
# .bzl files in this package (tools/build_defs/repo) are evaluated
# in a Starlark environment without "@_builtins" injection, and must not refer
# to symbols associated with build/workspace .bzl files

"""Code for interacting with git binary to get the file tree checked out at the specified revision.
"""

_GitRepoInfo = provider(
    doc = "Provider to organize precomputed arguments for calling git.",
    fields = {
        "directory": "Working directory path",
        "shallow": "Defines the depth of a fetch. Either empty, --depth=1, or --shallow-since=<>",
        "reset_ref": """Reference to use for resetting the git repository.
Either commit hash, tag or branch.""",
        "fetch_ref": """Reference for fetching.
Either commit hash, tag or branch.""",
        "remote": "URL of the git repository to fetch from.",
        "init_submodules": """If True, submodules update command will be called after fetching
and resetting to the specified reference.""",
        "recursive_init_submodules": """if True, all submodules will be updated recursively
after fetching and resetting the repo to the specified instance.""",
    },
)

def git_repo(ctx, directory):
    """ Fetches data from git repository and checks out file tree.

    Called by git_repository or new_git_repository rules.

    Args:
        ctx: Context of the calling rules, for reading the attributes.
        Please refer to the git_repository and new_git_repository rules for the description.
        directory: Directory where to check out the file tree.
    Returns:
        The struct with the following fields:
        commit: Actual HEAD commit of the checked out data.
        shallow_since: Actual date and time of the HEAD commit of the checked out data.
    """
    if ctx.attr.shallow_since:
        if ctx.attr.tag:
            fail("shallow_since not allowed if a tag is specified; --depth=1 will be used for tags")
        if ctx.attr.branch:
            fail("shallow_since not allowed if a branch is specified; --depth=1 will be used for branches")

    # Use shallow-since if given
    if ctx.attr.shallow_since:
        shallow = "--shallow-since=%s" % ctx.attr.shallow_since
    else:
        shallow = "--depth=1"

    reset_ref = ""
    fetch_ref = ""
    if ctx.attr.commit:
        reset_ref = ctx.attr.commit
        fetch_ref = ctx.attr.commit
    elif ctx.attr.tag:
        reset_ref = "tags/" + ctx.attr.tag
        fetch_ref = "tags/" + ctx.attr.tag + ":tags/" + ctx.attr.tag
    elif ctx.attr.branch:
        reset_ref = "origin/" + ctx.attr.branch
        fetch_ref = ctx.attr.branch + ":origin/" + ctx.attr.branch

    git_repo = _GitRepoInfo(
        directory = ctx.path(directory),
        shallow = shallow,
        reset_ref = reset_ref,
        fetch_ref = fetch_ref,
        remote = str(ctx.attr.remote),
        init_submodules = ctx.attr.init_submodules,
        recursive_init_submodules = ctx.attr.recursive_init_submodules,
    )

    _report_progress(ctx, git_repo)
    if ctx.attr.verbose:
        print("git.bzl: Cloning or updating %s repository %s using strip_prefix of [%s]" %
              (
                  " (%s)" % shallow if shallow else "",
                  ctx.name,
                  ctx.attr.strip_prefix if ctx.attr.strip_prefix else "None",
              ))

    _update(ctx, git_repo)
    ctx.report_progress("Recording actual commit")
    actual_commit = _get_head_commit(ctx, git_repo)
    shallow_date = _get_head_date(ctx, git_repo)

    return struct(commit = actual_commit, shallow_since = shallow_date)

def _git_version(ctx):
    """Gets the version of the Git executable."""
    command = ["git", "--version"]
    st = ctx.execute(command)
    if st.return_code != 0:
        _error(ctx.name, command, st.stderr)

    # The output of `git --version` is in the format:
    #
    #     git version <major>.<minor>.<revision>[ ...]
    #
    # The revision may be a non-integer, so it is not converted to an int. Any additional text
    # after <revision> is discarded.
    version_str = st.stdout.split(" ")[2].rstrip("\n")
    version_arr = version_str.split(".")
    return struct(
        major = int(version_arr[0]),
        minor = int(version_arr[1]),
        revision = version_arr[2],
        full_str = version_str,
    )

def _report_progress(ctx, git_repo, *, shallow_failed = False):
    warning = ""
    if shallow_failed:
        warning = " (shallow fetch failed, fetching full history)"
    ctx.report_progress("Cloning %s of %s%s" % (git_repo.reset_ref, git_repo.remote, warning))

def _update(ctx, git_repo):
    ctx.delete(git_repo.directory)

    init(ctx, git_repo)
    add_origin(ctx, git_repo, ctx.attr.remote)
    fetch(ctx, git_repo)
    reset(ctx, git_repo)
    clean(ctx, git_repo)

    if git_repo.recursive_init_submodules:
        ctx.report_progress("Updating submodules recursively")
        update_submodules(ctx, git_repo, recursive = True)
    elif git_repo.init_submodules:
        ctx.report_progress("Updating submodules")
        update_submodules(ctx, git_repo)

def init(ctx, git_repo):
    cl = ["git", "init", str(git_repo.directory)]
    st = ctx.execute(cl, environment = ctx.os.environ | _GIT_LOCAL_ENV_VARS)
    if st.return_code != 0:
        _error(ctx.name, cl, st.stderr)

def add_origin(ctx, git_repo, remote):
    _git(ctx, git_repo, "remote", "add", "origin", remote)

def fetch(ctx, git_repo):
    args = ["fetch", "origin", git_repo.fetch_ref]

    sparse_checkout_patterns_or_file = \
        getattr(ctx.attr, "sparse_checkout_patterns", None) or \
        getattr(ctx.attr, "sparse_checkout_file", None)
    if sparse_checkout_patterns_or_file:
        if _git_sparse_checkout_config(ctx, git_repo):
            # Use filter to disable downloading file contents until we set the `sparse-checkout` patterns.
            args.append("--filter=blob:none")
        else:
            print("WARNING: Sparse checkout is not supported. Doing a full checkout.")
            sparse_checkout_patterns_or_file = None

    st = _git_maybe_shallow(ctx, git_repo, *args)

    if sparse_checkout_patterns_or_file:
        _git_sparse_checkout(ctx, git_repo, sparse_checkout_patterns_or_file)

    if st.return_code == 0:
        return
    if ctx.attr.commit:
        # Perhaps uploadpack.allowReachableSHA1InWant or similar is not enabled on the server;
        # fall back to fetching all branches, tags, and history.
        # The semantics of --tags flag of git-fetch have changed in Git 1.9, from 1.9 it means
        # "everything that is already specified and all tags"; before 1.9, it used to mean
        # "ignore what is specified and fetch all tags".
        # The arguments below work correctly for both before 1.9 and after 1.9,
        # as we directly specify the list of references to fetch.
        _report_progress(ctx, git_repo, shallow_failed = True)
        _git(
            ctx,
            git_repo,
            "fetch",
            "origin",
            "refs/heads/*:refs/remotes/origin/*",
            "refs/tags/*:refs/tags/*",
        )
    else:
        _error(ctx.name, ["git"] + args, st.stderr)

def reset(ctx, git_repo):
    _git(ctx, git_repo, "reset", "--hard", git_repo.reset_ref)

def clean(ctx, git_repo):
    _git(ctx, git_repo, "clean", "-xdf")

def update_submodules(ctx, git_repo, recursive = False):
    if recursive:
        # "protocol.file.allow=always" allows the submodule command clone from a local directory.
        # It's necessary for Git 2.38.1 and assoicated backport versions.
        # See https://github.com/bazelbuild/bazel/issues/17040
        _git(ctx, git_repo, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive", "--checkout", "--force")
    else:
        _git(ctx, git_repo, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--checkout", "--force")

def _get_head_commit(ctx, git_repo):
    return _git(ctx, git_repo, "log", "-n", "1", "--pretty=format:%H")

def _get_head_date(ctx, git_repo):
    return _git(ctx, git_repo, "log", "-n", "1", "--pretty=format:%cd", "--date=raw")

def _git(ctx, git_repo, command, *args):
    start = [command]
    st = _execute(ctx, git_repo, start + list(args))
    if st.return_code != 0:
        _error(ctx.name, ["git"] + start + list(args), st.stderr)
    return st.stdout

def _git_maybe_shallow(ctx, git_repo, command, *args):
    start = [command]
    args_list = list(args)
    if git_repo.shallow:
        st = _execute(ctx, git_repo, start + [git_repo.shallow] + args_list)
        if st.return_code == 0:
            return st
    return _execute(ctx, git_repo, start + args_list)

def _git_sparse_checkout_config(ctx, git_repo):
    """Configures the repo for a sparse checkout.

    If the Git executable does not support sparse checkout, this function prints a warning and returns False.
    Otherwise, it returns True."""

    git_version = _git_version(ctx)

    # Sparse checkout was added in version 2.25.0.
    if git_version.major < 2 or (git_version.major == 2 and git_version.minor < 25):
        print("WARNING: Git v%s does not support sparse checkout." % (git_version.full_str))
        return False

    # Older versions of Git require this config to be set to the name of the promisor remote.
    config_command = ["config", "extensions.partialClone", "origin"]
    st = _execute(ctx, git_repo, config_command)
    if st.return_code != 0:
        _error(ctx.name, config_command, st.stderr)
    return True

def _git_sparse_checkout(ctx, git_repo, sparse_checkout_patterns_or_file):
    """Initialize the repo with patterns for a sparse checkout.

    Args:
        ctx: Context of the calling rules.
        git_repo: The Git repository to initialize for sparse checkout.
        sparse_checkout_patterns_or_file: Either a list of patterns or a Label for a sparse-checkout file.
    """

    # Note: `init` is deprecated, but needed for older versions of Git. This command may be removed
    # in future versions.
    init_command = ["sparse-checkout", "init", "--no-cone"]
    st = _execute(ctx, git_repo, init_command)
    if st.return_code != 0:
        _error(ctx.name, init_command, st.stderr)

    if type(sparse_checkout_patterns_or_file) == "list":
        sparse_checkout_patterns = sparse_checkout_patterns_or_file
        set_command = ["sparse-checkout", "set"] + sparse_checkout_patterns
        st = _execute(ctx, git_repo, set_command)
        if st.return_code != 0:
            _error(ctx.name, set_command, st.stderr)
    else:
        sparse_checkout_file = sparse_checkout_patterns_or_file
        link_name = str(git_repo.directory) + "/.git/info/sparse-checkout"
        ctx.delete(link_name)
        ctx.symlink(sparse_checkout_file, link_name)

# List of variables to unset when calling `git` to ensure no interference of
# operation. This is in the form of a dict that can be passed to `execute()`.
# This list is taken from the output of `git rev-parse --local-env-vars`
_GIT_LOCAL_ENV_VARS = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES": None,
    "GIT_CONFIG": None,
    "GIT_CONFIG_PARAMETERS": None,
    "GIT_CONFIG_COUNT": None,
    "GIT_OBJECT_DIRECTORY": None,
    "GIT_DIR": None,
    "GIT_WORK_TREE": None,
    "GIT_IMPLICIT_WORK_TREE": None,
    "GIT_GRAFT_FILE": None,
    "GIT_INDEX_FILE": None,
    "GIT_NO_REPLACE_OBJECTS": None,
    "GIT_REPLACE_REF_BASE": None,
    "GIT_PREFIX": None,
    "GIT_INTERNAL_SUPER_PREFIX": None,
    "GIT_SHALLOW_FILE": None,
    "GIT_COMMON_DIR": None,
}

def _execute(ctx, git_repo, args):
    # "core.fsmonitor=false" disables git from spawning a file system monitor which can cause hangs when cloning a lot.
    # See https://github.com/bazelbuild/bazel/issues/21438
    start = ["git", "-c", "core.fsmonitor=false"]
    return ctx.execute(
        start + args,
        environment = ctx.os.environ | _GIT_LOCAL_ENV_VARS,
        working_directory = str(git_repo.directory),
    )

def _error(name, command, stderr):
    command_text = " ".join([str(item).strip() for item in command])
    fail("error running '%s' while working with @%s:\n%s" % (command_text, name, stderr))
