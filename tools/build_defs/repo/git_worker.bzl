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
"""Code for interacting with git binary to get the file tree checked out at the specified revision.
"""

_GitRepoInfo = provider(
    doc = "Provider to organize precomputed arguments for calling git.",
    fields = {
        "directory": "Working directory path",
        "shallow": "Defines the depth of a fetch. Either empty, --depth=1, or --shallow-since=<>",
        "reset_ref": """Reference to use for resetting the git repository.
Either commit hash, tag or branch.""",
        "fetch_ref": """Reference for fetching. Can be empty (HEAD), tag or branch.
Can not be a commit hash, since typically it is forbidden by git servers.""",
        "remote": "URL of the git repository to fetch from.",
        "init_submodules": """If True, submodules update command will be called after fetching
and resetting to the specified reference.""",
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

    shallow = "--depth=1"
    if ctx.attr.commit:
        # We can not use the commit value in --shallow-since;
        # And since we are fetching HEAD in this case, we can not use --depth=1
        shallow = ""

    # Use shallow-since if given
    if ctx.attr.shallow_since:
        shallow = "--shallow-since=%s" % ctx.attr.shallow_since

    reset_ref = ""
    fetch_ref = ""
    if ctx.attr.commit:
        reset_ref = ctx.attr.commit
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
        remote = ctx.attr.remote,
        init_submodules = ctx.attr.init_submodules,
    )

    ctx.report_progress("Cloning %s of %s" % (reset_ref, ctx.attr.remote))
    if (ctx.attr.verbose):
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

def _update(ctx, git_repo):
    ctx.delete(git_repo.directory)

    init(ctx, git_repo)
    add_origin(ctx, git_repo, ctx.attr.remote)
    fetch(ctx, git_repo)
    reset(ctx, git_repo)
    clean(ctx, git_repo)

    if git_repo.init_submodules:
        ctx.report_progress("Updating submodules")
        update_submodules(ctx, git_repo)

def init(ctx, git_repo):
    cl = ["git", "init", str(git_repo.directory)]
    st = ctx.execute(cl, environment = ctx.os.environ)
    if st.return_code != 0:
        _error(ctx.name, cl, st.stderr)

def add_origin(ctx, git_repo, remote):
    _git(ctx, git_repo, "remote", "add", "origin", remote)

def fetch(ctx, git_repo):
    if not git_repo.fetch_ref:
        # We need to explicitly specify to fetch all branches and tags, otherwise only
        # HEAD-reachable is fetched.
        # The semantics of --tags flag of git-fetch have changed in Git 1.9, from 1.9 it means
        # "everything that is already specified and all tags"; before 1.9, it used to mean
        # "ignore what is specified and fetch all tags".
        # The arguments below work correctly for both before 1.9 and after 1.9,
        # as we directly specify the list of references to fetch.
        _git_maybe_shallow(
            ctx,
            git_repo,
            "fetch",
            "origin",
            "refs/heads/*:refs/remotes/origin/*",
            "refs/tags/*:refs/tags/*",
        )
    else:
        _git_maybe_shallow(ctx, git_repo, "fetch", "origin", git_repo.fetch_ref)

def reset(ctx, git_repo):
    _git(ctx, git_repo, "reset", "--hard", git_repo.reset_ref)

def clean(ctx, git_repo):
    _git(ctx, git_repo, "clean", "-xdf")

def update_submodules(ctx, git_repo):
    _git(ctx, git_repo, "submodule", "update", "--init", "--checkout", "--force")

def _get_head_commit(ctx, git_repo):
    return _git(ctx, git_repo, "log", "-n", "1", "--pretty=format:%H")

def _get_head_date(ctx, git_repo):
    return _git(ctx, git_repo, "log", "-n", "1", "--pretty=format:%cd", "--date=raw")

def _git(ctx, git_repo, command, *args):
    start = ["git", command]
    st = _execute(ctx, git_repo, start + list(args))
    if st.return_code != 0:
        _error(ctx.name, start + list(args), st.stderr)
    return st.stdout

def _git_maybe_shallow(ctx, git_repo, command, *args):
    start = ["git", command]
    args_list = list(args)
    if git_repo.shallow:
        st = _execute(ctx, git_repo, start + [git_repo.shallow] + args_list)
        if st.return_code == 0:
            return
    st = _execute(ctx, git_repo, start + args_list)
    if st.return_code != 0:
        _error(ctx.name, start + args_list, st.stderr)

def _execute(ctx, git_repo, args):
    return ctx.execute(
        args,
        environment = ctx.os.environ,
        working_directory = str(git_repo.directory),
    )

def _error(name, command, stderr):
    fail("error running '%s' while working with @%s:\n%s" % (" ".join(command).strip(), name, stderr))
