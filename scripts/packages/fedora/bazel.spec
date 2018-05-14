Name: bazel
Version: devel
Release: 1
Summary: Correct, reproducible, and fast builds for everyone.
URL: https://bazel.build
License: Apache License, v2.0

Source0: {bazel}
Source1: {bazel-real}
Source2: {bazel.bazelrc}
Source3: {bazel-complete.bash}

Requires: java-1.8.0-openjdk-headless

%description
Bazel is a build tool that builds code quickly and reliably. It is used to build the majority of Google's software, and thus it has been designed to handle build problems present in Google's development environment.

%global _enable_debug_package 0
%global debug_package %{nil}
%global __os_install_post /usr/lib/rpm/brp-compress %{nil}

%prep

%build

%install
export DONT_STRIP=1
mkdir -p %{buildroot}%{_bindir}/
install -m 755 {bazel} %{buildroot}%{_bindir}/bazel
install -m 755 {bazel-real} %{buildroot}%{_bindir}/bazel-real
mkdir -p %{buildroot}%{_sysconfdir}/
install -m 644 {bazel.bazelrc} %{buildroot}%{_sysconfdir}/bazel.bazelrc
mkdir -p %{buildroot}%{_sysconfdir}/bash_completion.d/
install -m 644 {bazel-complete.bash} %{buildroot}%{_sysconfdir}/bash_completion.d/bazel

%files
%{_bindir}/bazel
%{_bindir}/bazel-real
%{_sysconfdir}/bazel.bazelrc
%{_sysconfdir}/bash_completion.d/bazel

%changelog
* Fri Jan 27 2017 jcater@google.com - devel-1
- Initial package version released.

