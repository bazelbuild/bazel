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
    It assumes the parameters `name`, `build_file`, `build_file_content`,
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
        ctx.file("WORKSPACE", ctx.read(ctx.attr.workspace_file))
    elif ctx.attr.workspace_file_content:
        ctx.file("WORKSPACE", ctx.attr.workspace_file_content)
    else:
        ctx.file("WORKSPACE", "workspace(name = \"{name}\")\n".format(name = ctx.name))

    if ctx.attr.build_file:
        ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))
    elif ctx.attr.build_file_content:
        ctx.file("BUILD.bazel", ctx.attr.build_file_content)

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def _use_native_patch(patch_args):
    """If patch_args only contains -p<NUM> options, we can use the native patch implementation."""
    for arg in patch_args:
        if not arg.startswith("-p"):
            return False
    return True

def patch(ctx, patches = None, patch_cmds = None, patch_cmds_win = None, patch_tool = None, patch_args = None):
    """Implementation of patching an already extracted repository.

    This rule is inteded to be used in the implementation function of
    a repository rule. Ifthe parameters `patches`, `patch_tool`,
    `patch_args`, `patch_cmds` and `patch_cmds_win` are not specified
    then they are taken from `ctx.attr`.

    Args:
      ctx: The repository context of the repository rule calling this utility
        function.
      patches: The patch files to apply. List of strings, Labels, or paths.
      patch_cmds: Bash commands to run for patching, passed one at a
        time to bash -c. List of strings
      patch_cmds_win: Powershell commands to run for patching, passed
        one at a time to powershell /c. List of strings. If the
        boolean value of this parameter is false, patch_cmds will be
        used and this parameter will be ignored.
      patch_tool: Path of the patch tool to execute for applying
        patches. String.
      patch_args: Arguments to pass to the patch tool. List of strings.

    """
    bash_exe = ctx.os.environ["BAZEL_SH"] if "BAZEL_SH" in ctx.os.environ else "bash"
    powershell_exe = ctx.os.environ["BAZEL_POWERSHELL"] if "BAZEL_POWERSHELL" in ctx.os.environ else "powershell.exe"

    if patches == None and hasattr(ctx.attr, "patches"):
        patches = ctx.attr.patches
    if patches == None:
        patches = []

    if patch_cmds == None and hasattr(ctx.attr, "patch_cmds"):
        patch_cmds = ctx.attr.patch_cmds
    if patch_cmds == None:
        patch_cmds = []

    if patch_cmds_win == None and hasattr(ctx.attr, "patch_cmds_win"):
        patch_cmds_win = ctx.attr.patch_cmds_win
    if patch_cmds_win == None:
        patch_cmds_win = []

    if patch_tool == None and hasattr(ctx.attr, "patch_tool"):
        patch_tool = ctx.attr.patch_tool
    if not patch_tool:
        patch_tool = "patch"
        native_patch = True
    else:
        native_patch = False

    if patch_args == None and hasattr(ctx.attr, "patch_args"):
        patch_args = ctx.attr.patch_args
    if patch_args == None:
        patch_args = []

    if len(patches) > 0 or len(patch_cmds) > 0:
        ctx.report_progress("Patching repository")

    if native_patch and _use_native_patch(patch_args):
        if patch_args:
            strip = int(patch_args[-1][2:])
        else:
            strip = 0
        for patchfile in patches:
            ctx.patch(patchfile, strip)
    else:
        for patchfile in patches:
            command = "{patchtool} {patch_args} < {patchfile}".format(
                patchtool = patch_tool,
                patchfile = ctx.path(patchfile),
                patch_args = " ".join([
                    "'%s'" % arg
                    for arg in patch_args
                ]),
            )
            st = ctx.execute([bash_exe, "-c", command])
            if st.return_code:
                fail("Error applying patch %s:\n%s%s" %
                     (str(patchfile), st.stderr, st.stdout))

    if _is_windows(ctx) and patch_cmds_win:
        for cmd in patch_cmds_win:
            st = ctx.execute([powershell_exe, "/c", cmd])
            if st.return_code:
                fail("Error applying patch command %s:\n%s%s" %
                     (cmd, st.stdout, st.stderr))
    else:
        for cmd in patch_cmds:
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
    if not native.existing_rule(name):
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
    contents = ctx.read(filename)

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
        if line.startswith("#"):
            # Comments start with #. Ignore these lines.
            continue
        elif macdef:
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

def use_netrc(netrc, urls, patterns):
    """compute an auth dict from a parsed netrc file and a list of URLs

    Args:
      netrc: a netrc file already parsed to a dict, e.g., as obtained from
        read_netrc
      urls: a list of URLs.
      patterns: optional dict of url to authorization patterns

    Returns:
      dict suitable as auth argument for ctx.download; more precisely, the dict
      will map all URLs where the netrc file provides login and password to a
      dict containing the corresponding login, password and optional authorization pattern,
      as well as the mapping of "type" to "basic" or "pattern".
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
        if host in patterns:
            auth_dict = {
                "type": "pattern",
                "pattern": patterns[host],
            }

            if "login" in authforhost:
                auth_dict["login"] = authforhost["login"]

            if "password" in authforhost:
                auth_dict["password"] = authforhost["password"]

            auth[url] = auth_dict
        elif "login" in authforhost and "password" in authforhost:
            auth[url] = {
                "type": "basic",
                "login": authforhost["login"],
                "password": authforhost["password"],
            }

    return auth
