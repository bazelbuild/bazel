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

"""Interacts with the `git` binary to check out a file tree at a specified revision.

Supports two modes, selected by the BAZEL_GIT_REPOSITORY_CACHE environment variable:

* Cached: a bare repository under the cache directory is (re)used as a shared
  object store, and each checkout is a `git worktree` created from it. Concurrent
  accesses to the shared cache are serialized with `flock`.
* Uncached: a plain clone is created directly in the destination directory.

Submodules are checked out explicitly (breadth-first) in the cached mode and via
`git submodule update` in the uncached mode.
"""

def _join(*segments):
    """Joins path segments with "/".

    Dependency-free replacement for skylib's `paths.join`: this file lives in
    @bazel_tools, which cannot load @bazel_skylib.
    """
    return "/".join([segment.rstrip("/") for segment in segments if segment])

_GitRepoInfo = provider(
    doc = "Provider to organize precomputed arguments for calling git.",
    fields = {
        "cache_dir": """Path to a bare git repository, shared across fetches and located under
$BAZEL_GIT_REPOSITORY_CACHE, used as an object cache so repeated fetches of the same remote
avoid re-downloading; the checkout itself is a git worktree created from it.""",
        "cache_lockfile": """Path to the lock file (cache_dir + ".lockfile") passed to `flock` to
serialize concurrent access to the shared cache_dir across parallel fetches.""",
        "directory": "Working directory path",
        "shallow": "Defines the depth of a fetch. Either empty, --depth=1, or --shallow-since=<>",
        "reset_ref": """Reference to use for resetting the git repository.
Either commit hash, tag, branch name, or default branch.""",
        "fetch_ref": """Reference for fetching.
Either commit hash, tag, branch name, or default branch.""",
        "remote": "URL of the git repository to fetch from.",
        "init_submodules": """If True, submodules update command will be called after fetching
and resetting to the specified reference.""",
        "recursive_init_submodules": """if True, all submodules will be updated recursively
after fetching and resetting the repo to the specified instance.""",
    },
)

def _trace(ctx, message):
    """Prints `message` only when the rule's `verbose` attribute is set."""
    if ctx.attr.verbose:
        print(message)  # buildifier: disable=print

def git_repo(ctx, directory):
    """ Fetches data from git repository and checks out file tree.

    Called by git_repository rule.

    Args:
        ctx: Context of the calling rules, for reading the attributes.
        Please refer to the git_repository rule for the description.
        directory: Directory where to check out the file tree.
    Returns:
        The struct with the following fields:
        commit: Actual HEAD commit of the checked out data.
        shallow_since: Actual date and time of the HEAD commit of the checked out data.
    """

    # Use shallow-since if given
    if ctx.attr.shallow_since:
        if ctx.attr.tag:
            fail("shallow_since not allowed if a tag is specified; --depth=1 will be used for tags")
        if ctx.attr.branch:
            fail("shallow_since not allowed if a branch is specified; --depth=1 will be used for branches")
        shallow = "--shallow-since=%s" % ctx.attr.shallow_since
    else:
        shallow = "--depth=1"

    if ctx.attr.commit:
        reset_ref = ctx.attr.commit
        fetch_ref = ctx.attr.commit
    elif ctx.attr.tag:
        reset_ref = "tags/" + ctx.attr.tag
        fetch_ref = "tags/" + ctx.attr.tag + ":tags/" + ctx.attr.tag
    elif ctx.attr.branch:
        reset_ref = "origin/" + ctx.attr.branch
        fetch_ref = ctx.attr.branch + ":origin/" + ctx.attr.branch
    else:
        reset_ref = "origin/HEAD"
        fetch_ref = "HEAD:refs/remotes/origin/HEAD"

    cache_root = ctx.os.environ.get("BAZEL_GIT_REPOSITORY_CACHE")
    if cache_root:
        cache_dir = _join(cache_root, "{}.{}".format(ctx.path(ctx.attr.remote).basename, str(hash(ctx.attr.remote))))
        cache_lockfile = cache_dir + ".lockfile"
    else:
        cache_dir = None
        cache_lockfile = None

    git_repo = _GitRepoInfo(
        cache_dir = cache_dir,
        cache_lockfile = cache_lockfile,
        directory = ctx.path(directory),
        shallow = shallow,
        reset_ref = reset_ref,
        fetch_ref = fetch_ref,
        remote = str(ctx.attr.remote),
        init_submodules = ctx.attr.init_submodules,
        recursive_init_submodules = ctx.attr.recursive_init_submodules,
    )

    _trace(ctx, "git.bzl: Cloning or updating %s repository %s using strip_prefix of [%s] into %s%s" %
                (
                    "(%s)" % shallow if shallow else "",
                    ctx.name,
                    ctx.attr.strip_prefix if ctx.attr.strip_prefix else "None",
                    directory,
                    " cached on %s" % cache_dir if cache_dir else "",
                ))

    _update(ctx, git_repo)
    ctx.report_progress("Recording actual commit")
    actual_commit = _get_head_commit(ctx, git_repo).stdout
    shallow_date = _get_head_date(ctx, git_repo).stdout

    # Remove all .git subfolders from the worktree so downstream rules
    # never see nested git metadata (e.g. from submodules).
    ctx.report_progress("Removing .git subfolders from worktree")
    _remove_git_dirs(ctx, git_repo.directory)

    # Clean up all worktree information from the cache directory, to not leave
    # any residual states. Not using `git worktree prune` since we don't want
    # locked worktrees to block this operation.
    if git_repo.cache_dir:
        ctx.delete(_join(git_repo.cache_dir, "worktrees", git_repo.directory.basename))

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
    """Reports a "Cloning <ref> of <remote>" progress line, noting when a shallow fetch fell back to full history."""
    warning = ""
    if shallow_failed:
        warning = " (shallow fetch failed, fetching full history)"
    ctx.report_progress("Cloning %s of %s%s" % (git_repo.reset_ref, git_repo.remote, warning))

# Maximum supported git submodule nesting depth. Real submodule graphs are only
# a few levels deep, so this ceiling is never reached in practice.
_MAX_SUBMODULE_DEPTH = 100

def _update(ctx, git_repo):
    """Checks out `git_repo` and all of its submodules into their worktrees.

    Clears the target directory, then walks the repo and any (nested) submodules
    breadth-first, fetching and resetting each one.
    """
    ctx.delete(git_repo.directory)

    # Breadth-first checkout of the repo and any (nested) submodules it declares.
    # Starlark cannot iterate a queue that grows during traversal, so we process
    # one nesting level at a time: `frontier` holds the repos at the current
    # depth, and each pass collects the submodules it discovers into
    # `next_frontier`. The inner loop is bounded by the known size of the current
    # level, so the number of submodules per level is unbounded; only the nesting
    # depth is capped (see _MAX_SUBMODULE_DEPTH).
    frontier = [git_repo]
    for _ in range(_MAX_SUBMODULE_DEPTH):
        if not frontier:
            break
        next_frontier = []
        for git_repo in frontier:
            _report_progress(ctx, git_repo)

            if git_repo.cache_dir:
                # Skip the fetch when the commit is already cached. `cat-file -e` avoids
                # the `^{...}` peel syntax, which ctx.execute mangles on Windows, and git
                # reads need a valid working directory, so run from the cache (the worktree
                # is not checked out yet); an absent cache simply reports a miss.
                cache_wd = str(git_repo.cache_dir) if ctx.path(git_repo.cache_dir).exists else str(git_repo.directory)
                st = _git(ctx, git_repo, False, False, ["cat-file", "-e", git_repo.fetch_ref], working_directory = cache_wd)
                if st.return_code != 0:
                    init(ctx, git_repo)
                    add_origin(ctx, git_repo)
                    fetch(ctx, git_repo)

                # A commit fetched by raw SHA is reachable only through the worktree HEAD,
                # which is removed after each build; pin it with a ref so the shared cache
                # keeps it and `git gc` cannot prune it. Ref-based fetches (a ":" in
                # fetch_ref) already create reachable refs.
                if ":" not in git_repo.fetch_ref:
                    _git(ctx, git_repo, False, True, ["update-ref", "refs/bazelcache/" + git_repo.fetch_ref, git_repo.fetch_ref])

                reset(ctx, git_repo)

                # Get the submodule paths and their URLs
                submodules = _gitmodules_values(ctx, git_repo, "path")
                urls = _gitmodules_values(ctx, git_repo, "url")

                # Queue each submodule for checkout at the next nesting level
                if git_repo.init_submodules or git_repo.recursive_init_submodules:
                    for submodule_path, submodule_url in zip(submodules, urls):
                        # `git ls-tree HEAD <path>` prints "<mode> <type> <object>\t<path>";
                        # extract <object>, the commit the parent repo pins this submodule to.
                        st = _git(ctx, git_repo, True, True, ["ls-tree", "HEAD", submodule_path])
                        sha = st.stdout.split(" ")[2].split("\t")[0]

                        if "://" not in submodule_url:
                            # It's a relative path. Sometimes submodules use it when they're part of the same server as the parent's repository.
                            submodule_url = join_url_path(git_repo.remote, submodule_url)

                        submodule_cache_dir = _join(ctx.os.environ.get("BAZEL_GIT_REPOSITORY_CACHE"), "{}.{}".format(ctx.path(submodule_url).basename, str(hash(submodule_url))))
                        submodule_cache_lockfile = submodule_cache_dir + ".lockfile"

                        submodule_repo = _GitRepoInfo(
                            cache_dir = submodule_cache_dir,
                            cache_lockfile = submodule_cache_lockfile,
                            directory = git_repo.directory.get_child(submodule_path),
                            shallow = git_repo.shallow,
                            reset_ref = sha,
                            fetch_ref = sha,
                            remote = submodule_url,
                            init_submodules = False,
                            recursive_init_submodules = git_repo.recursive_init_submodules,
                        )

                        _trace(ctx, "Found submodule:\npath={}\nurl={}\nsha={}".format(submodule_path, submodule_url, sha))

                        # Extend the list of repos to check out at the next level
                        next_frontier.append(submodule_repo)

            else:
                init(ctx, git_repo)
                add_origin(ctx, git_repo)
                fetch(ctx, git_repo)
                reset(ctx, git_repo)

                if git_repo.recursive_init_submodules:
                    ctx.report_progress("Updating submodules recursively")
                    update_submodules(ctx, git_repo, recursive = True)
                elif git_repo.init_submodules:
                    ctx.report_progress("Updating submodules")
                    update_submodules(ctx, git_repo)

        # Advance to the next nesting level: the submodules discovered while
        # processing the current frontier become the frontier for the next pass.
        # When none were queued (the common case) this empties the frontier and
        # the outer loop stops on its next iteration.
        frontier = next_frontier

    if frontier:
        fail("git submodule nesting exceeds the maximum supported depth of %d" % _MAX_SUBMODULE_DEPTH)

def _gitmodules_values(ctx, git_repo, key):
    """Returns the value of `.gitmodules` entries matching `key` (e.g. "path" or "url"), one per submodule.

    Returns an empty list when the repo declares no submodules (no `.gitmodules`).
    """
    st = _git(ctx, git_repo, False, False, ["config", "--file", ".gitmodules", "--get-regexp", key])
    if st.return_code != 0:
        return []

    # Each matching line is "submodule.<name>.<key> <value>"; keep the <value>.
    return [line.split(" ")[1] for line in st.stdout.strip().split("\n") if line.strip()]

def _sparse_checkout_source(ctx):
    """Returns the configured sparse-checkout patterns list or sparse-checkout file, or None if neither is set."""
    return getattr(ctx.attr, "sparse_checkout_patterns", None) or getattr(ctx.attr, "sparse_checkout_file", None)

def init(ctx, git_repo):
    """Initializes the git repository.

    Creates a bare repository at the shared cache when caching is enabled,
    otherwise a normal repository in the worktree.

    Args:
        ctx: Repository context.
        git_repo: _GitRepoInfo describing the repository to initialize.
    """
    if git_repo.cache_dir:
        cl = ["init", "--bare", git_repo.cache_dir]
    else:
        cl = ["init", str(git_repo.directory)]

    _git(ctx, git_repo, False, True, cl)

def add_origin(ctx, git_repo):
    """Adds the `origin` remote pointing at `git_repo.remote`.

    Args:
        ctx: Repository context.
        git_repo: _GitRepoInfo describing the repository.
    """
    cl = ["remote", "add", "origin", git_repo.remote]
    st = _git(ctx, git_repo, False, False, cl)

    # A pre-existing `origin` (e.g. from a reused cache repo) is fine; any other failure is fatal.
    if st.return_code != 0 and "already exists" not in st.stderr:
        _error(ctx.name, ["git"] + cl, st.stderr)

def fetch(ctx, git_repo):
    """Fetches `git_repo.fetch_ref` from origin, honoring sparse-checkout and shallow settings.

    Falls back to fetching all branches and tags when a commit-specific fetch is
    refused by the server (e.g. reachable-SHA1 uploads not enabled).

    Args:
        ctx: Repository context.
        git_repo: _GitRepoInfo describing the repository.
    """
    args = ["origin"]

    sparse_checkout_patterns_or_file = _sparse_checkout_source(ctx)
    if sparse_checkout_patterns_or_file:
        if _git_sparse_checkout_config(ctx, git_repo):
            # Use filter to disable downloading file contents until we set the `sparse-checkout` patterns.
            args.append("--filter=blob:none")
        else:
            print("WARNING: Sparse checkout is not supported. Doing a full checkout.")
            sparse_checkout_patterns_or_file = None

    args.extend(["--", git_repo.fetch_ref])
    st = _git_fetch_maybe_shallow(ctx, git_repo, args)

    if sparse_checkout_patterns_or_file and not git_repo.cache_dir:
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
        _git(ctx, git_repo, False, True, ["fetch", "origin", "--", "refs/heads/*:refs/remotes/origin/*", "refs/tags/*:refs/tags/*"])
    else:
        _error(ctx.name, ["git", "fetch"] + args, st.stderr)

def reset(ctx, git_repo):
    """Materializes `git_repo.reset_ref` into the worktree.

    With a shared cache this adds a detached `git worktree` (configuring sparse
    checkout before the files are written); otherwise it hard-resets the clone
    and removes untracked files.

    Args:
        ctx: Repository context.
        git_repo: _GitRepoInfo describing the repository.
    """
    if git_repo.cache_dir:
        # Create worktree without checking out files first
        _git(ctx, git_repo, False, True, ["worktree", "add", "-f", "-f", "--no-checkout", "--detach", str(git_repo.directory), git_repo.reset_ref])

        # Configure sparse checkout after worktree is created but before checkout
        sparse_checkout_patterns_or_file = _sparse_checkout_source(ctx)
        if sparse_checkout_patterns_or_file:
            _git_sparse_checkout(ctx, git_repo, sparse_checkout_patterns_or_file)

        # Now checkout with sparse patterns applied
        _git(ctx, git_repo, True, True, ["--work-tree=" + str(git_repo.directory), "checkout", git_repo.reset_ref])
    else:
        _git(ctx, git_repo, True, True, ["reset", "--hard", git_repo.reset_ref])
        _git(ctx, git_repo, True, True, ["clean", "-xdf"])

def update_submodules(ctx, git_repo, recursive = False):
    """Initializes and checks out submodules with `git submodule update` (uncached checkout path).

    Args:
        ctx: Repository context.
        git_repo: _GitRepoInfo describing the repository.
        recursive: If True, also initialize submodules of submodules (`--recursive`).
    """

    # "protocol.file.allow=always" allows the submodule command clone from a local directory.
    # It's necessary for Git 2.38.1 and assoicated backport versions.
    # See https://github.com/bazelbuild/bazel/issues/17040
    args = ["-c", "protocol.file.allow=always", "submodule", "update", "--init", "--checkout", "--force"]
    if recursive:
        args.append("--recursive")
    _git(ctx, git_repo, True, True, args)

def _get_head_commit(ctx, git_repo):
    """Returns the git result whose stdout is the checked-out HEAD commit hash."""
    return _git(ctx, git_repo, True, True, ["log", "-n", "1", "--pretty=format:%H"])

def _get_head_date(ctx, git_repo):
    """Returns the git result whose stdout is the HEAD commit date in raw ("<epoch> <tz>") form."""
    return _git(ctx, git_repo, True, True, ["log", "-n", "1", "--pretty=format:%cd", "--date=raw"])

def _remove_git_dirs(ctx, directory):
    """Finds and removes all .git files and directories under the given directory."""
    st = ctx.execute(["find", str(directory), "-name", ".git"], timeout = 60)
    if st.return_code == 0:
        for entry in st.stdout.strip().split("\n"):
            if entry:
                ctx.delete(entry)

def _git_fetch_maybe_shallow(ctx, git_repo, args):
    """Runs `git fetch <args>`, attempting the configured shallow fetch first and falling back to a full fetch on failure."""
    if git_repo.shallow:
        st = _git(ctx, git_repo, False, False, ["fetch", git_repo.shallow] + args)
        if st.return_code == 0:
            return st
    ctx.report_progress("Shallow fetch failed for {}, trying full fetch".format(git_repo.remote))
    return _git(ctx, git_repo, False, False, ["fetch"] + args)

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
    return _git(ctx, git_repo, False, True, ["config", "extensions.partialClone", "origin"]).return_code == 0

def _git_sparse_checkout(ctx, git_repo, sparse_checkout_patterns_or_file):
    """Initialize the repo with patterns for a sparse checkout.

    Args:
        ctx: Context of the calling rules.
        git_repo: The Git repository to initialize for sparse checkout.
        sparse_checkout_patterns_or_file: Either a list of patterns or a Label for a sparse-checkout file.
    """

    if git_repo.cache_dir:
        work_tree = ["--work-tree=" + str(git_repo.directory)]
        link_name = _join(git_repo.cache_dir, "worktrees", git_repo.directory.basename, "info/sparse-checkout")
    else:
        work_tree = []
        link_name = str(git_repo.directory.get_child(".git/info/sparse-checkout"))

    # Note: `init` is deprecated, but needed for older versions of Git. This command may be removed
    # in future versions.
    _git(ctx, git_repo, True, True, work_tree + ["sparse-checkout", "init", "--no-cone"])

    if type(sparse_checkout_patterns_or_file) == "list":
        _git(ctx, git_repo, True, True, work_tree + ["sparse-checkout", "set"] + sparse_checkout_patterns_or_file)
    elif git_repo.cache_dir:
        # For a linked worktree the sparse-checkout file lives in the shared bare
        # repo's per-worktree admin directory, which is outside this repository's
        # output directory; ctx.symlink refuses to write there. Overwrite the file
        # that `sparse-checkout init` just created with a subprocess copy, which is
        # not restricted to the repository directory.
        ctx.execute(["cp", "-f", str(ctx.path(sparse_checkout_patterns_or_file)), link_name])
    else:
        ctx.delete(link_name)
        ctx.symlink(sparse_checkout_patterns_or_file, link_name)

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

def _git(ctx, git_repo, exec_in_worktree, fail_is_error, args, working_directory = None):
    """Runs a single git command for `git_repo` and returns the execution result.

    Commands run from `git_repo.directory` unless `working_directory` overrides it,
    and are serialized with `flock` on `git_repo.cache_lockfile` whenever a shared
    cache is in use.

    Args:
        ctx: Repository context used to execute the command.
        git_repo: _GitRepoInfo describing the repository being operated on.
        exec_in_worktree: If True, operate on the checked-out worktree (its own
            `.git`). If False and a cache is configured, operate on the bare cache
            repository via `--git-dir`.
        fail_is_error: If True, abort the build (via _error) when git exits
            non-zero; if False, return the result for the caller to inspect.
        args: The git subcommand and its arguments, without the leading "git".
        working_directory: Directory to run git in; defaults to `git_repo.directory`.

    Returns:
        The `ctx.execute` result, exposing `return_code`, `stdout` and `stderr`.
    """
    git_dir = ["--git-dir={}".format(git_repo.cache_dir)] if git_repo.cache_dir and not exec_in_worktree else []

    flock = ["flock", "-x", git_repo.cache_lockfile] if git_repo.cache_lockfile else []

    if working_directory == None:
        working_directory = str(git_repo.directory)

    # "core.fsmonitor=false" disables git from spawning a file system monitor which can cause hangs when cloning a lot.
    # See https://github.com/bazelbuild/bazel/issues/21438
    cl = flock + ["git", "-c", "core.fsmonitor=false"] + git_dir + args
    _trace(ctx, " ".join(cl))
    st = ctx.execute(cl, environment = ctx.os.environ | _GIT_LOCAL_ENV_VARS, working_directory = working_directory, timeout = 7200)
    if fail_is_error and st.return_code != 0:
        _error(git_repo.remote, ["git"] + args, st.stderr)
    return st

def _error(name, command, stderr):
    """Aborts the build with a formatted message describing the failed git `command` for repo `name`."""
    command_text = " ".join([str(item).strip() for item in command])
    fail("error running '%s' while working with @%s:\n%s" % (command_text, name, stderr))

def join_url_path(base_url, rel_path):
    """Resolves a relative submodule path against a base URL, normalizing "." and ".." segments.

    Used for `.gitmodules` entries that specify a submodule location relative to
    the parent repository's URL instead of an absolute one.

    Args:
        base_url: Absolute URL of the parent repository (must contain "://").
        rel_path: Relative path from the parent repository to the submodule.

    Returns:
        The absolute URL of the submodule.
    """

    # Split URL at '://'
    scheme_split = base_url.split("://", 1)
    if len(scheme_split) != 2:
        fail("Invalid URL: %s" % base_url)
    scheme, rest = scheme_split

    # Split rest into netloc and path
    if "/" in rest:
        netloc, base_path = rest.split("/", 1)
        base_parts = base_path.split("/")
    else:
        netloc = rest
        base_parts = []

    rel_parts = rel_path.split("/")

    # Normalize path
    path_parts = base_parts
    for part in rel_parts:
        if part == "..":
            if path_parts:
                path_parts.pop()
        elif part != "." and part != "":
            path_parts.append(part)

    normalized_path = "/".join(path_parts)
    return scheme + "://" + netloc + "/" + normalized_path
