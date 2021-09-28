package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import org.junit.Test;

public class CcToolchainInputsTransitionFactoryTest extends BuildViewTestCase {

  @Test
  public void testTargetTransitionForInputsEnabled_usesTargetPlatform() throws Exception {
    scratch.file(
        "a/BUILD",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "filegroup(",
        "    name = 'all_files',",
        "    srcs = ['a.txt'],",
        ")",
        "cc_toolchain(",
        "    name = 'toolchain',",
        "    all_files = ':all_files',",
        "    ar_files = ':all_files',",
        "    as_files = ':all_files',",
        "    compiler_files = ':all_files',",
        "    compiler_files_without_includes = ':all_files',",
        "    dwp_files = ':all_files',",
        "    linker_files = ':all_files',",
        "    strip_files = ':all_files',",
        "    objcopy_files = ':all_files',",
        "    toolchain_identifier = 'does-not-matter',",
        "    toolchain_config = ':does-not-matter-config',",
        "    target_transition_for_inputs = True,",
        ")",
        "cc_toolchain_config(name = 'does-not-matter-config')"
    );

    scratch.file("a/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);

    ConfiguredTarget toolchainTarget = getConfiguredTarget("//a:toolchain");
    assertThat(toolchainTarget).isNotNull();

    ConfiguredTarget allFiles = getDirectPrerequisite(toolchainTarget, "//a:all_files");
    assertThat(allFiles).isNotNull();

    assertThat(
        getTargetConfiguration().equalsOrIsSupersetOf(getConfiguration(allFiles))).isTrue();

    CoreOptions coreOptions = getConfiguration(allFiles).getOptions().get(CoreOptions.class);
    assertThat(coreOptions).isNotNull();
    assertThat(coreOptions.isHost).isFalse();
    assertThat(coreOptions.isExec).isFalse();
  }

  @Test
  public void testTargetTransitionForInputsDisabled_usesExecPlatform() throws Exception {
    scratch.file(
        "a/BUILD",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "filegroup(",
        "    name = 'all_files',",
        "    srcs = ['a.txt'],",
        ")",
        "cc_toolchain(",
        "    name = 'toolchain',",
        "    all_files = ':all_files',",
        "    ar_files = ':all_files',",
        "    as_files = ':all_files',",
        "    compiler_files = ':all_files',",
        "    compiler_files_without_includes = ':all_files',",
        "    dwp_files = ':all_files',",
        "    linker_files = ':all_files',",
        "    strip_files = ':all_files',",
        "    objcopy_files = ':all_files',",
        "    toolchain_identifier = 'does-not-matter',",
        "    toolchain_config = ':does-not-matter-config',",
        "    target_transition_for_inputs = False,",
        ")",
        "cc_toolchain_config(name = 'does-not-matter-config')"
    );

    scratch.file("a/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);

    ConfiguredTarget toolchainTarget = getConfiguredTarget("//a:toolchain");
    assertThat(toolchainTarget).isNotNull();

    ConfiguredTarget allFiles = getDirectPrerequisite(toolchainTarget, "//a:all_files");
    assertThat(allFiles).isNotNull();

    CoreOptions coreOptions = getConfiguration(allFiles).getOptions().get(CoreOptions.class);
    assertThat(coreOptions).isNotNull();
    assertThat(coreOptions.isHost).isFalse();
    assertThat(coreOptions.isExec).isTrue();
  }


}
