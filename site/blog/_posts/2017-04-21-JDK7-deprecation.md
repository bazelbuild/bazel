---
layout: posts
title: JDK7 deprecation
---

The Bazel team has been maintaining a separate, stripped-down build of Bazel
that runs with JDK 7. The 0.5.1 release will no longer provide this special
version.

To address the problem of JDK 8 not being available on some machines, starting
with version 0.5.0, our installer will embed a JDK by default.

If you have any concerns, please reach out to
[bazel-discuss@googlegroups.com](mailto:bazel-discuss@googlegroups.com).

__Recap:__

__0.5.0__:

  * `bazel-0.5.0-installer.sh`: default version, with embedded JDK.
  * `bazel-0.5.0-without-jdk-installer.sh`: version without embedded JDK.
  * `bazel-0.5.0-jdk7-installer.sh`: last release compatible with JDK 7.

__0.5.1__:

  * `bazel-0.5.1-installer.sh`: default version, with embedded JDK.
  * `bazel-0.5.1-without-jdk-installer.sh`: version without embedded JDK.

__Migration path:__

If you are currently using the Bazel with JDK 7, then starting with version
0.5.0 you must start using the default installer.

If you are currently using the default installer and do not want to use a
version with embedded JDK, then use the `-without-jdk` version.

__Note:__

Homebrew and debian packages do not contain the embedded JDK. This change only
affects the shell installers.

__Thanks:__

Thanks everybody for bearing with all the JDK 7 related issues, including the
Java team at Google, in particular
[Liam Miller-Cushon](https://github.com/cushon).

Special thanks to [Philipp Wollermann](https://github.com/philwo) who made this
new installer possible.


