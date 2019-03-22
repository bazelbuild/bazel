load("@bazel_tools//tools/build_defs/repo:utils.bzl", "remove_dir")

GitRepo = provider(
    doc = "TODO",
    fields = {
        "directory": "TODO",
        "shallow": "TODO",
        "reset_ref": "TODO",
        "fetch_ref": "TODO",
        "remote": "TODO",
        "init_submodules": "TODO",
    },
)

def git_repo(ctx, directory):
    if ctx.attr.shallow_since:
        if ctx.attr.tag:
            fail("shallow_since not allowed if a tag is specified; --depth=1 will be used for tags")
        if ctx.attr.branch:
            fail("shallow_since not allowed if a branch is specified; --depth=1 will be used for branches")

    shallow = "--depth=1"
    if ctx.attr.shallow_since:
        shallow = "--shallow-since=%s" % ctx.attr.shallow_since
    if ctx.attr.commit:
        shallow = ""

    reset_ref = ""
    fetch_ref = ""
    if ctx.attr.commit:
        reset_ref = ctx.attr.commit
    elif ctx.attr.tag:
        reset_ref = "tags/" + ctx.attr.tag
        fetch_ref = "tags/" + ctx.attr.tag + ":tags/" + ctx.attr.tag
    elif ctx.attr.branch:
        reset_ref = "origin/" + ctx.attr.branch
        fetch_ref = ctx.attr.branch

    git_repo = GitRepo(
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
    actual_commit = _get_head_commit(ctx, git_repo)
    shallow_date = _get_head_date(ctx, git_repo)

    return struct(commit = actual_commit, shallow_since = shallow_date)

def _update(ctx, git_repo):
    remove_dir(ctx, git_repo.directory)

    init(ctx, git_repo)
    add_origin(ctx, git_repo, ctx.attr.remote)
    fetch(ctx, git_repo)
    reset(ctx, git_repo)
    clean(ctx, git_repo)

    if git_repo.init_submodules:
        update_submodules(ctx, git_repo)

def init(ctx, git_repo):
    cl = ["git", "init", "%s" % git_repo.directory]
    st = ctx.execute(cl, environment = ctx.os.environ)
    if st.return_code != 0:
        _error(ctx.name, cl, st.stderr)

def add_origin(ctx, git_repo, remote):
    _git(ctx, git_repo, "remote", "add", "origin", remote)

def fetch(ctx, git_repo):
    _git_maybe_shallow(ctx, git_repo, "fetch", "origin", git_repo.fetch_ref)

def reset(ctx, git_repo):
    _git_maybe_shallow(ctx, git_repo, "reset", "--hard", git_repo.reset_ref, "--")

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
    st = _execute(ctx, git_repo, start + ["'%s'" % git_repo.shallow] + args_list)
    if st.return_code != 0:
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
