Project: /_project.yaml
Book: /_book.yaml

# Command-Line Completion

{% include "_buttons.html" %}

You can enable command-line completion (also known as tab-completion) in Bash
and Zsh. This lets you tab-complete command names, flags names and flag values,
and target names.

## Bash {:#bash}

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

*   Via [bootstrapping](/install/compile-source), then:
    1.  Emit the completion script into a file:

        ```
        bazel help completion bash > bazel-complete.bash
        ```
    2.  Do one of the following:
        *   Copy this file to your completion directory, if you have
            one.

            Example: on Ubuntu this is the `/etc/bash_completion.d` directory
        *   Copy it somewhere on your local disk, such as to `$HOME`, and
            source the completion file from Bash's RC file.

            Add a line similar to the one below to your `~/.bashrc` (on Ubuntu)
            or `~/.bash_profile` (on macOS), using the path to your completion
            file's absolute path:

            ```
            source /path/to/bazel-complete.bash
            ```

## Zsh {:#zsh}

Bazel comes with a Zsh completion script.

If you installed Bazel:

*   From the APT repository, then you're done -- the Zsh completion script is
    already installed in `/usr/share/zsh/vendor-completions`.

    > If you have a heavily customized `.zshrc` and the autocomplete
    > does not function, try one of the following solutions:
    >
    > Add the following to your `.zshrc`:
    >
    >    ```
    >     zstyle :compinstall filename '/home/tradical/.zshrc'
    >
    >     autoload -Uz compinit
    >     compinit
    >    ```
    >
    > or
    >
    > Follow the instructions
    > [here](https://stackoverflow.com/questions/58331977/bazel-tab-auto-complete-in-zsh-not-working)
    >
    > If you are using `oh-my-zsh`, you may want to install and enable
    > the `zsh-autocomplete` plugin. If you'd prefer not to, use one of the
    > solutions described above.

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
