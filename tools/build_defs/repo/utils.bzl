# Copyright 2018 The Bazel Authors. All rights reserved.
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
"""Utils for manipulating external repositories, once fetched.

### Setup

These utility are intended to be used by other repository rules. They
can be loaded as follows.

```python
load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "workspace_and_buildfile",
    "patch",
    "update_attrs",
)
```
"""

def workspace_and_buildfile(ctx):
    """Utility function for writing WORKSPACE and, if requested, a BUILD file.

    This rule is inteded to be used in the implementation function of a
    repository rule.
    It assumes the parameters `name`, `build_file`, `build_file_contents`,
    `workspace_file`, and `workspace_file_content` to be
    present in `ctx.attr`, the latter four possibly with value None.

    Args:
      ctx: The repository context of the repository rule calling this utility
        function.
    """
    if ctx.attr.build_file and ctx.attr.build_file_content:
        ctx.fail("Only one of build_file and build_file_content can be provided.")

    if ctx.attr.workspace_file and ctx.attr.workspace_file_content:
        ctx.fail("Only one of workspace_file and workspace_file_content can be provided.")

    if ctx.attr.workspace_file:
        ctx.delete("WORKSPACE")
        ctx.symlink(ctx.attr.workspace_file, "WORKSPACE")
    elif ctx.attr.workspace_file_content:
        ctx.delete("WORKSPACE")
        ctx.file("WORKSPACE", ctx.attr.workspace_file_content)
    else:
        ctx.file("WORKSPACE", "workspace(name = \"{name}\")\n".format(name = ctx.name))

    if ctx.attr.build_file:
        ctx.delete("BUILD.bazel")
        ctx.symlink(ctx.attr.build_file, "BUILD.bazel")
    elif ctx.attr.build_file_content:
        ctx.delete("BUILD.bazel")
        ctx.file("BUILD.bazel", ctx.attr.build_file_content)

def patch(ctx):
    """Implementation of patching an already extracted repository.

    This rule is inteded to be used in the implementation function of a
    repository rule. It assuumes that the parameters `patches`, `patchtool`,
    `patch_args`, and `patch_cmds` are present in `ctx.attr`.

    Args:
      ctx: The repository context of the repository rule calling this utility
        function.
    """
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    if len(ctx.attr.patches) > 0 or len(ctx.attr.patch_cmds) > 0:
        ctx.report_progress("Patching repository")
    for patchfile in ctx.attr.patches:
        command = "{patchtool} {patch_args} < {patchfile}".format(
            patchtool = ctx.attr.patch_tool,
            patchfile = ctx.path(patchfile),
            patch_args = " ".join([
                "'%s'" % arg
                for arg in ctx.attr.patch_args
            ]),
        )
        st = ctx.execute([bash_exe, "-c", command])
        if st.return_code:
            fail("Error applying patch %s:\n%s%s" %
                 (str(patchfile), st.stderr, st.stdout))
    for cmd in ctx.attr.patch_cmds:
        st = ctx.execute([bash_exe, "-c", cmd])
        if st.return_code:
            fail("Error applying patch command %s:\n%s%s" %
                 (cmd, st.stdout, st.stderr))

def update_attrs(orig, keys, override):
    """Utility function for altering and adding the specified attributes to a particular repository rule invocation.

     This is used to make a rule reproducible.

    Args:
        orig: dict of actually set attributes (either explicitly or implicitly)
            by a particular rule invocation
        keys: complete set of attributes defined on this rule
        override: dict of attributes to override or add to orig

    Returns:
        dict of attributes with the keys from override inserted/updated
    """
    result = {}
    for key in keys:
        if getattr(orig, key) != None:
            result[key] = getattr(orig, key)
    result["name"] = orig.name
    result.update(override)
    return result

def maybe(repo_rule, name, **kwargs):
    """Utility function for only adding a repository if it's not already present.

    This is to implement safe repositories.bzl macro documented in
    https://docs.bazel.build/versions/master/skylark/deploying.html#dependencies.

    Args:
        repo_rule: repository rule function.
        name: name of the repository to create.
        **kwargs: remaining arguments that are passed to the repo_rule function.

    Returns:
        Nothing, defines the repository when needed as a side-effect.
    """
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)

def read_netrc(ctx, filename):
    """Utility function to parse at least a basic .netrc file.

    Args:
      ctx: The repository context of the repository rule calling this utility
        function.
      filename: the name of the .netrc file to read

    Returns:
      dict mapping a machine names to a dict with the information provided
      about them
    """

    # We have to first symlink into the current repository, as ctx.read only
    # allows read from the output directory. Alternatively, we could use
    # ctx.execute() to call cat(1).
    ctx.symlink(filename, ".netrc")
    contents = ctx.read(".netrc")
    ctx.delete(".netrc")

    # Parse the file. This is mainly a token-based update of a simple state
    # machine, but we need to keep the line structure to correctly determine
    # the end of a `macdef` command.
    netrc = {}
    currentmachinename = None
    currentmachine = {}
    macdef = None
    currentmacro = ""
    cmd = None
    for line in contents.splitlines():
        if macdef:
            # as we're in a macro, just determine if we reached the end.
            if line:
                currentmacro += line + "\n"
            else:
                # reached end of macro, add it
                currentmachine[macdef] = currentmacro
                macdef = None
                currentmacro = ""
        else:
            # Essentially line.split(None) which starlark does not support.
            tokens = [
                w.strip()
                for w in line.split(" ")
                if len(w.strip()) > 0
            ]
            for token in tokens:
                if cmd:
                    # we have a command that expects another argument
                    if cmd == "machine":
                        # a new machine definition was provided, so save the
                        # old one, if present
                        if not currentmachinename == None:
                            netrc[currentmachinename] = currentmachine
                        currentmachine = {}
                        currentmachinename = token
                    elif cmd == "macdef":
                        macdef = "macdef %s" % (token,)
                        # a new macro definition; the documentation says
                        # "its contents begin with the next .netrc line [...]",
                        # so should there really be tokens left in the current
                        # line, they're not part of the macro.

                    else:
                        currentmachine[cmd] = token
                    cmd = None
                elif token in [
                    "machine",
                    "login",
                    "password",
                    "account",
                    "macdef",
                ]:
                    # command takes one argument
                    cmd = token
                elif token == "default":
                    # defines the default machine; again, store old machine
                    if not currentmachinename == None:
                        netrc[currentmachinename] = currentmachine

                    # We use the empty string for the default machine, as that
                    # can never be a valid hostname ("default" could be, in the
                    # default search domain).
                    currentmachinename = ""
                    currentmachine = {}
                else:
                    fail("Unexpected token '%s' while reading %s" %
                         (token, filename))
    if not currentmachinename == None:
        netrc[currentmachinename] = currentmachine
    return netrc

def use_netrc(netrc, urls):
    """compute an auth dict from a parsed netrc file and a list of URLs

    Args:
      netrc: a netrc file already parsed to a dict, e.g., as obtained from
        read_netrc
      urls: a list of URLs.

    Returns:
      dict suitable as auth argument for ctx.download; more precisely, the dict
      will map all URLs where the netrc file provides login and password to a
      dict containing the corresponding login and passwored, as well as the
      mapping of "type" to "basic"
    """
    auth = {}
    for url in urls:
        schemerest = url.split("://", 1)
        if len(schemerest) < 2:
            continue
        if not (schemerest[0] in ["http", "https"]):
            # For other protocols, bazel currently does not support
            # authentication. So ignore them.
            continue
        host = schemerest[1].split("/")[0].split(":")[0]
        if not host in netrc:
            continue
        authforhost = netrc[host]
        if "login" in authforhost and "password" in authforhost:
            auth[url] = {
                "type": "basic",
                "login": authforhost["login"],
                "password": authforhost["password"],
            }
    return auth
