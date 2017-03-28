---
layout: documentation
title: Installing Bazel
---

# Installing Bazel

See the instructions for installing Bazel on:

*   [Ubuntu Linux (16.04, 15.10, and 14.04)](install-ubuntu.md)
*   [Mac OS X](install-os-x.md)
*   [Windows (highly experimental)](install-windows.md)

For other platforms, you can try to [compile from source](install-compile-source.md).

Required Java version:

*   Java JDK 8 or later ([JDK 7](#jdk7) is still supported
    but deprecated).

Extras:

*   [Bash completion](#bash)
*   [zsh completion](#zsh)

For more information on using Bazel, see
[Getting Started with Bazel](getting-started.html).


## <a name="jdk7"></a>Using Bazel with JDK 7 (deprecated)

Bazel version _0.1.0_ runs without any change with JDK 7. However, future
version will stop supporting JDK 7 when our CI cannot build for it anymore.
The installer for JDK 7 for Bazel versions after _0.1.0_ is labeled
<pre>
./bazel-<em>version</em>-jdk7-installer-<em>os</em>.sh
</pre>
If you wish to use JDK 7, follow the same steps as for JDK 8 but with the _jdk7_ installer or using a different APT repository as described [here](#1-add-bazel-distribution-uri-as-a-package-source-one-time-setup).

## <a name="bash"></a>Getting bash completion

Bazel comes with a bash completion script, which the installer copies into the
`bin` directory. If you ran the installer with `--user`, this will be
`$HOME/.bazel/bin`. If you ran the installer as root, this will be
`/usr/local/bazel/bin`.

Copy the `bazel-complete.bash` script to your completion folder
(`/etc/bash_completion.d` directory under Ubuntu). If you don't have a
completion folder, you can copy it wherever suits you and insert
`source /path/to/bazel-complete.bash` in your `~/.bashrc` file (under OS X, put
it in your `~/.bash_profile` file).

If you built Bazel from source, the bash completion target is in the `//scripts`
package:

1. Build it with Bazel: `bazel build //scripts:bazel-complete.bash`.
2. Copy the script `bazel-bin/scripts/bazel-complete.bash` to one of the
   locations described above.

## <a name="zsh"></a>Getting zsh completion

Bazel also comes with a zsh completion script. To install it:

1. Add this script to a directory on your $fpath:

    ```
    fpath[1,0]=~/.zsh/completion/
    mkdir -p ~/.zsh/completion/
    cp scripts/zsh_completion/_bazel ~/.zsh/completion
    ```

    You may have to call `rm -f ~/.zcompdump; compinit`
    the first time to make it work.

2. Optionally, add the following to your .zshrc.

    ```
    # This way the completion script does not have to parse Bazel's options
    # repeatedly.  The directory in cache-path must be created manually.
    zstyle ':completion:*' use-cache on
    zstyle ':completion:*' cache-path ~/.zsh/cache
    ```
