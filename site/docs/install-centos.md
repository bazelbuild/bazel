---
layout: documentation
title: Installing Bazel on CentOS
---

# <a name="centos"></a>Install Bazel on CentOS

Supported CentOS Linux platforms:

*   CentOS 7.3
*   CentOS 7.4

Install Bazel on CentOS using one of the following methods:

*   [Use our custom Bazel COPR repository (experimental)](#install-on-centos)
*   [Compile Bazel from source (experimental)](install-compile-source.md)

Bazel comes with two completion scripts. After installing Bazel, you can:

*   access the [bash completion script](install.md)
*   install the [zsh completion script](install.md)

## <a name="install-on-centos"></a> Using Bazel custom COPR repository (experimental)

### 1. Install Bazel COPR repository (one time setup)

```bash
sudo curl -L https://copr.fedorainfracloud.org/coprs/sdake/bazel/repo/epel-7/sdake-bazel-epel-7.repo -o /etc/yum.repos.d/sdake-bazel-epel-7.repo
```

### 2. Install Bazel

```bash
sudo yum install bazel
```

### 3. Update Bazel

Once installed, you can upgrade to a newer version of Bazel with:

```bash
sudo yum update bazel
```
