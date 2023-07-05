# Toolchain Resolution Implementation Details

Several SkyFunction classes implement the [toolchain resolution] process:

1.  [`RegisteredToolchainsFunction`] and
    [`RegisteredExecutionPlatformsFunction`] find available toolchains and
    execution platforms, based on the current configuration and WORKSPACE file.

1.  [`SingleToolchainResolutionFunction`] resolves a single toolchain type for
    every execution platform. That is, for every execution platform it finds the
    best registered toolchain to use based on the following criteria:

    1.  Make sure the toolchain and target platform are compatible, by checking
        the `target_compatible_with` attribute.
    1.  Make sure the toolchain and execution platform are compatible, by
        checking the `exec_compatible_with` attribute.
    1.  If multiple toolchains are left, choose the highest-priority one (the
        one that was registered first).

1.  [`ToolchainResolutionFunction`] calls `SingleToolchainResolutionFunction`
    for each requested toolchain type, and then determines the best execution
    platform to use.

    1.  First, remove any execution platform that does not have a valid
        toolchain for each requested toolchain type.
    2.  If multiple execution platforms are left, choose the highest-priority
        one (the one that was registered first).
        1.  If the execution platform is already set by the toolchain
            transition, it will be selected first as described below.

As discussed in [Toolchains and Configurations][toolchains_and_configurations],
the dependency from a target to a toolchain uses a special configuration that
forces the execution platform to be the same for both. Despite the name
"toolchain transition", this is not implemented as a configuration transition,
but instead as a special subclass of [`ConfiguredTargetKey`], called
[`ToolchainDependencyConfiguredTargetKey`].  In addition to the other data in
`ConfiguredTargetKey`, this subclass also holds the label of the execution
platform. When `ToolchainResolutionFunction` is considering which execution
platform to use, if the forced execution platform from the
`ToolchainDependencyConfiguredTargetKey` is valid, it will be used even if it is
not the highest-priority.

**Note:** If the forced execution platform is not valid (because there are no
valid toolchains, or because of execution constraints from the rule or target),
then the highest-priority valid execution platform will be used instead.

**TODO:** Update this to discuss execution groups.

[toolchain resolution]: https://bazel.build/extending/toolchains
[toolchains_and_configurations]: https://bazel.build/extending/toolchains#toolchains_and_configurations
[`ConfiguredTargetKey`]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/ConfiguredTargetKey.java
[`RegisteredExecutionPlatformsFunction`]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/toolchains/RegisteredExecutionPlatformsFunction.java
[`RegisteredToolchainsFunction`]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/toolchains/RegisteredToolchainsFunction.java
[`SingleToolchainResolutionFunction`]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/toolchains/SingleToolchainResolutionFunction.java;
[`ToolchainDependencyConfiguredTargetKey`]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/ConfiguredTargetKey.java?q=symbol%3A%5Cbcom.google.devtools.build.lib.skyframe.ConfiguredTargetKey.ToolchainDependencyConfiguredTargetKey%5Cb%20case%3Ayes
[`ToolchainResolutionFunction`]: https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/toolchains/ToolchainResolutionFunction.java

