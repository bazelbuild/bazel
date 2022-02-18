---
layout: documentation
title: Installing Bazel on openSUSE
category: getting-started
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/install/suse" style="color: #0000EE;">https://bazel.build/install/suse</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


# Installing Bazel on openSUSE Tumbleweed & Leap

This page describes how to install Bazel on openSUSE Tumbleweed and Leap.

NOTE: The Bazel team does not officially maintain openSUSE support. For issues
using Bazel on openSUSE please file a ticket at [bugzilla.opensuse.org](https://bugzilla.opensuse.org/).

Packages are provided for openSUSE Tumbleweed and Leap. You can find all
available Bazel versions via openSUSE's [software search](https://software.opensuse.org/search?utf8=%E2%9C%93&baseproject=ALL&q=bazel).

The commands below must be run either via `sudo` or while logged in as `root`.

## Installing Bazel on openSUSE

Run the following commands to install the package. If you need a specific
version, you can install it via the specific `bazelXXX` package, otherwise,
just `bazel` is enough:

```bash
# Install the latest Bazel version.
zypper install bazel

# Alternatively: Install Bazel 1.2.
zypper install bazel1.2
```
