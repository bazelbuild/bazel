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

load(":utils.bzl", "join_paths")

_GitRepoInfo = provider(
    doc = "Provider to organize precomputed arguments for calling git.",
    fields = {
        "cache_dir": "Repository cached directory path",
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

    if ctx.os.environ.get("BAZEL_GIT_REPOSITORY_CACHE"):
        cache_dir = join_paths(ctx.os.environ.get("BAZEL_GIT_REPOSITORY_CACHE"), str(hash(ctx.attr.remote)))
    else:
        cache_dir = None

    git_repo = _GitRepoInfo(
        cache_dir = cache_dir,
        directory = ctx.path(directory),
        shallow = shallow,
        reset_ref = reset_ref,
        fetch_ref = fetch_ref,
        remote = str(ctx.attr.remote),
        init_submodules = ctx.attr.init_submodules,
        recursive_init_submodules = ctx.attr.recursive_init_submodules,
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

    # Clean up all worktree information from the cache directory, to not leave
    # any residual states. Not using `git worktree prune` since we don't want
    # locked worktrees to block this operation.
    ctx.delete("{}/worktrees".format(cache_dir))

    return struct(commit = actual_commit, shallow_since = shallow_date)

def _if_debug(cond, st, what = "Action"):
    "Print if 'cond'."
    if cond:
        print("{} returned {}\n{}\n----\n{}".format(what, st.return_code, st.stdout, st.stderr))

def _setup_cache(ctx, git_repo):
    # Create cache directory
    cl = ["mkdir", "-p", git_repo.cache_dir]
    st = ctx.execute(
        cl,
        environment = ctx.os.environ,
    )
    _if_debug(cond = ctx.attr.verbose, st = st, what = " ".join(cl))
    if st.return_code:
        _error(ctx.name, cl, st.stderr)

    # Init git cache directory
    cl = ["git", "init", "--bare"]
    st = ctx.execute(
        cl,
        environment = ctx.os.environ,
        working_directory = git_repo.cache_dir,
    )
    _if_debug(cond = ctx.attr.verbose, st = st, what = " ".join(cl))

def _get_repository_from_cache(ctx, git_repo):
    ctx.delete(join_paths(git_repo.cache_dir, "worktrees"))
    ctx.delete(git_repo.directory)

    cl = ["git", "worktree", "add", str(git_repo.directory), git_repo.reset_ref]
    st = ctx.execute(
        cl,
        environment = ctx.os.environ,
        working_directory = git_repo.cache_dir,
    )
    _if_debug(cond = ctx.attr.verbose, st = st, what = " ".join(cl))

    # This fails on the first run for a specific ref, but return the result and
    # don't stop the execution here; we'll need to know that it fails in order
    # to fall back to fetching from the remote repository.
    return st

def _populate_cache(ctx, git_repo):
    # Fetch w/ shallow and w/ ref
    cl = ["git", "fetch", git_repo.shallow, git_repo.remote, git_repo.reset_ref]
    st = ctx.execute(
        cl,
        environment = ctx.os.environ,
        working_directory = git_repo.cache_dir,
    )
    _if_debug(cond = ctx.attr.verbose, st = st, what = " ".join(cl))
    if st.return_code == 0:
        return

    # If above fails, try to fetch w/o shallow, and w/ ref
    cl = ["git", "fetch", git_repo.remote, git_repo.reset_ref]
    st = ctx.execute(
        cl,
        environment = ctx.os.environ,
        working_directory = git_repo.cache_dir,
    )
    _if_debug(cond = ctx.attr.verbose, st = st, what = " ".join(cl))
    if st.return_code == 0:
        return

    # If above fails, try to fetch w/ shallow, and w/o ref
    cl = ["git", "fetch", git_repo.shallow, git_repo.remote]
    st = ctx.execute(
        cl,
        environment = ctx.os.environ,
        working_directory = git_repo.cache_dir,
    )
    _if_debug(cond = ctx.attr.verbose, st = st, what = " ".join(cl))
    if st.return_code == 0:
        return

    # If above fails, try to fetch w/o shallow, and w/o ref
    cl = ["git", "fetch", git_repo.remote]
    st = ctx.execute(
        cl,
        environment = ctx.os.environ,
        working_directory = git_repo.cache_dir,
    )
    _if_debug(cond = ctx.attr.verbose, st = st, what = " ".join(cl))

    if st.return_code:
        _error(ctx.name, cl, st.stderr)

def _is_git_worktree_available(ctx):
    st = ctx.execute(
        ["git", "worktree", "--help"],
        environment = ctx.os.environ,
    )
    return st.return_code == 0

def _update(ctx, git_repo):
    ctx.delete(git_repo.directory)

    # Using git repository cache
    if git_repo.cache_dir and _is_git_worktree_available(ctx):
        _setup_cache(ctx, git_repo)
        st = _get_repository_from_cache(ctx, git_repo)
        if st.return_code:
            if ctx.attr.verbose:
                print("{} not found in cache. Fetching from remote...".format(git_repo.reset_ref))
            _populate_cache(ctx, git_repo)
            st = _get_repository_from_cache(ctx, git_repo)
            if st.return_code:
                fail("Error checking out worktree {}:\n{}".format(ctx.name, st.stderr))
    else:
        # Not using git repository cache
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
    st = ctx.execute(cl, environment = ctx.os.environ)
    if st.return_code != 0:
        _error(ctx.name, cl, st.stderr)

def add_origin(ctx, git_repo, remote):
    _git(ctx, git_repo, "remote", "add", "origin", remote)

def fetch(ctx, git_repo):
    args = ["fetch", "origin", git_repo.fetch_ref]
    st = _git_maybe_shallow(ctx, git_repo, *args)
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
        _git(ctx, git_repo, "submodule", "update", "--init", "--recursive", "--checkout", "--force")
    else:
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
            return st
    return _execute(ctx, git_repo, start + args_list)

def _execute(ctx, git_repo, args):
    return ctx.execute(
        args,
        environment = ctx.os.environ,
        working_directory = str(git_repo.directory),
    )

def _error(name, command, stderr):
    command_text = " ".join([str(item).strip() for item in command])
    fail("error running '%s' while working with @%s:\n%s" % (command_text, name, stderr))
