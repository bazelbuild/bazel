---
layout: documentation
title: Toolchain Resolution Implementation Details
---

# Toolchain Resolution Implementation Details


**Note:** This section is intended for Bazel developers, and is not needed by
rule authors.

Several SkyFunction classes implement the [toolchain resolution][Toolchains] process:

1.  [`RegisteredToolchainsFunction`][RegisteredToolchainsFunction] and
    [`RegisteredExecutionPlatformsFunction`][RegisteredExecutionPlatformsFunction]
    find available toolchains and execution platforms, based on the current
    configuration and WORKSPACE file.

1.  [`SingleToolchainResolutionFunction`][SingleToolchainResolutionFunction]
    resolves a single toolchain type for every execution platform. That is, for
    every execution platform it finds the best registered toolchain to use based
    on the following criteria:

    1.  Make sure the toolchain and target platform are compatible, by checking
        the `target_compatible_with` attribute.
    1.  Make sure the toolchain and execution platform are compatible, by
        checking the `exec_compatible_with` attribute.
    1.  If multiple toolchains are left, choose the highest-priority one (the
        one that was registered first).

1.  [`ToolchainResolutionFunction`][ToolchainResolutionFunction] calls
    `SingleToolchainResolutionFunction` for each requested toolchain type, and
    then determines the best execution platform to use.

    1.  First, remove any execution platform that does not have a valid
        toolchain for each requested toolchain type.
    2.  If multiple execution platforms are left, choose the highest-priority
        one (the one that was registered first).
        1.  If the execution platform is already set by the toolchain
            transition, it will be selected first as described below.

As discussed in [Toolchains and Configurations][Toolchains and Configurations],
the dependency from a target to a toolchain uses a special configuration that
forces the execution platform to be the same for both. Despite the name
"toolchain transition", this is not implemented as a configuration
transition, but instead as a special subclass of
[`ConfiguredTargetKey`][ConfiguredTargetKey], called
[`ToolchainDependencyConfiguredTargetKey`][ToolchainDependencyConfiguredTargetKey].
In addition to the other data in `ConfiguredTargetKey`, this subclass also holds
the label of the execution platform. When `ToolchainResolutionFunction` is
considering which execution platform to use, if the forced execution platform
from the `ToolchainDependencyConfiguredTargetKey` is valid, it will be used even
if it is not the highest-priority.

**Note:** If the forced execution platform is not valid (because there are no
valid toolchains, or because of execution constraints from the rule or target),
then the highest-priority valid execution platform will be used instead.

[ConfiguredTargetKey]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/ConfiguredTargetKey.java
[RegisteredExecutionPlatformsFunction]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/RegisteredExecutionPlatformsFunction.java
[RegisteredToolchainsFunction]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/RegisteredToolchainsFunction.java
[SingleToolchainResolutionFunction]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/SingleToolchainResolutionFunction.java
[ToolchainDependencyConfiguredTargetKey]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/ConfiguredTargetKey.java;bpv=1;bpt=1;l=164?ss=bazel&q=ConfiguredTargetKey&gsn=ToolchainDependencyConfiguredTargetKey&gs=kythe%3A%2F%2Fgithub.com%2Fbazelbuild%2Fbazel%3Flang%3Djava%3Fpath%3Dcom.google.devtools.build.lib.skyframe.ConfiguredTargetKey.ToolchainDependencyConfiguredTargetKey%2336c7e68f8cd5ea0b5a21b3769e63e6b2d489b9ca8c6f79798839e7f40cf2a19e
[ToolchainResolutionFunction]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/ToolchainResolutionFunction.java
[Toolchains]: toolchains.html
[Toolchains and Configurations]: toolchains.html#toolchains-and-configurations
