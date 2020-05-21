---
layout: documentation
title: Command-line completion
---

# Command-line completion

You can enable command-line completion (also known as tab-completion) in Bash
and Zsh. This lets you tab-complete command names, flags names and flag values,
and target names.

<h2 id="bash">Bash</h2>

Bazel comes with a Bash completion script.

If you installed Bazel:

*   From the APT repository, then you're done -- the Bash completion script is
    already installed in `/etc/bash_completion.d`.

*   From Homebrew, then you're done -- the Bash completion script is
    already installed in `$(brew --prefix)/etc/bash_completion.d`.

*   From the installer downloaded from GitHub, then:
    1.  Locate the absolute path of the completion file. The installer copied it
        to the `bin` directory.

        Example: if you ran the installer with `--user`, this will be
        `$HOME/.bazel/bin`. If you ran the installer as root, this will be
        `/usr/local/lib/bazel/bin`.
    2.  Do one of the following:
        *   Either copy this file to your completion directory (if you have
            one).

            Example: on Ubuntu this is the `/etc/bash_completion.d` directory.
        *   Or source the completion file from Bash's RC file.

            Add a line similar to the one below to your `~/.bashrc` (on Ubuntu)
            or `~/.bash_profile` (on macOS), using the path to your completion
            file's absolute path:

            ```
            source /path/to/bazel-complete.bash
            ```

*   Via [bootstrapping](install-compile-source.html), then:
    1.  Build the completion script:

        ```
        bazel build //scripts:bazel-complete.bash
        ```
    2.  The completion file is built under
        `bazel-bin/scripts/bazel-complete.bash`.

        Do one of the following:
        *   Either copy this file to your completion directory (if you have
            one).

            Example: on Ubuntu this is the `/etc/bash_completion.d` directory
        *   Or copy it somewhere on your local disk, e.g. to `$HOME`, and
            source the completion file from Bash's RC file.

            Add a line similar to the one below to your `~/.bashrc` (on Ubuntu)
            or `~/.bash_profile` (on macOS), using the path to your completion
            file's absolute path:

            ```
            source /path/to/bazel-complete.bash
            ```

<h2 id="zsh">Zsh</h2>

Bazel comes with a Zsh completion script.

If you installed Bazel:

*   From the APT repository, then you're done -- the Zsh completion script is
    already installed in `/usr/share/zsh/vendor-completions`.

*   From Homebrew, then you're done -- the Zsh completion script is
    already installed in `$(brew --prefix)/share/zsh/site-functions`.

*   From the installer downloaded from GitHub, then:
    1.  Locate the absolute path of the completion file. The installer copied it
        to the `bin` directory.

        Example: if you ran the installer with `--user`, this will be
        `$HOME/.bazel/bin`. If you ran the installer as root, this will be
        `/usr/local/lib/bazel/bin`.

    2.  Add this script to a directory on your `$fpath`:

        ```
        fpath[1,0]=~/.zsh/completion/
        mkdir -p ~/.zsh/completion/
        cp /path/from/above/step/_bazel ~/.zsh/completion
        ```

        You may have to call `rm -f ~/.zcompdump; compinit`
        the first time to make it work.

    3.  Optionally, add the following to your .zshrc.

        ```
        # This way the completion script does not have to parse Bazel's options
        # repeatedly.  The directory in cache-path must be created manually.
        zstyle ':completion:*' use-cache on
        zstyle ':completion:*' cache-path ~/.zsh/cache
        ```
