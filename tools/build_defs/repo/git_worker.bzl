GitRepo = provider(
    doc = "TODO",
    fields = {
        "directory": "TODO",
        "shallow": "TODO",
        "ref": "TODO",
        "fetch_head": "TODO",
    },
)

def git_repo(ctx):
    shallow = "--depth=1"
    if ctx.attr.shallow_since:
        shallow = "--shallow-since=%s" % ctx.attr.shallow_since
    if ctx.attr.commit:
        shallow = ""

    ref = "HEAD"
    if ctx.attr.commit:
        ref = ctx.attr.commit
    elif ctx.attr.tag:
        ref = "tags/" + ctx.attr.tag
    elif ctx.attr.branch:
        ref = ctx.attr.branch

    directory = str(ctx.path("."))
    if ctx.attr.strip_prefix:
        directory = directory + "-t m p"

    return GitRepo(
        directory = directory,
        shallow = shallow,
        ref = ref,
        fetch_head = ctx.attr.commit,
    )

def init(ctx, git_repo):
    cl = ["git", "init", "%s" % git_repo.directory]
    st = ctx.execute(cl, environment = ctx.os.environ)
    if st.return_code != 0:
        fail("error with %s %s:\n%s" % (" ".join(cl), ctx.name, st.stderr))

def add_origin(ctx, git_repo, remote):
    _git(ctx, git_repo, "remote", "add", "origin", remote)

def ensure_at_ref(ctx, git_repo):
    if not reset(ctx, git_repo, True):
        fetch(ctx, git_repo)
        reset(ctx, git_repo, False)
    clean(ctx, git_repo)

def fetch(ctx, git_repo):
    ref = git_repo.ref + ":" + git_repo.ref
    if git_repo.fetch_head:
        ref = ""
    _git_maybe_shallow(ctx, git_repo, "fetch", False, "origin", ref)

def reset(ctx, git_repo, silent):
    return _git_maybe_shallow(ctx, git_repo, "reset", silent, "--hard", git_repo.ref)

def clean(ctx, git_repo):
    _git(ctx, git_repo, "clean", "-xdf")

def update_submodules(ctx, git_repo):
    _git(ctx, git_repo, "submodule", "update", "--init", "--checkout", "--force")

def get_head_commit(ctx, git_repo):
    return _git(ctx, git_repo, "log", "-n", "1", "--pretty=format:%H")

def get_head_date(ctx, git_repo):
    return _git(ctx, git_repo, "log", "-n", "1", "--pretty=format:%cd", "--date=raw")

def _git(ctx, git_repo, command, *args):
    start = ["git", command]
    st = ctx.execute(
        start + list(args),
        environment = ctx.os.environ,
        working_directory = git_repo.directory,
    )
    if st.return_code != 0:
        fail("error with %s %s:\n%s" % (" ".join(start + list(args)), ctx.name, st.stderr))
    return st.stdout

def _git_maybe_shallow(ctx, git_repo, command, silent, *args):
    start = ["git", command]
    st = ctx.execute(
        start + ["'%s'" % git_repo.shallow] + list(args),
        environment = ctx.os.environ,
        working_directory = git_repo.directory,
    )
    if st.return_code != 0:
        st = ctx.execute(
            start + list(args),
            environment = ctx.os.environ,
            working_directory = git_repo.directory,
        )
        if st.return_code != 0:
            if not silent:
                fail("error with %s %s:\n%s" % (" ".join(start + list(args)), ctx.name, st.stderr))
            return False
    return True
