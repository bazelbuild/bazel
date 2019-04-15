package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.ConstraintValueLookupUtil.InvalidConstraintValueException;
import com.google.devtools.build.lib.skyframe.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.ToolchainResolutionFunction.NoMatchingPlatformException;
import com.google.devtools.build.lib.skyframe.ToolchainResolutionFunction.UnresolvedToolchainsException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UnloadedToolchainContext} and {@link ToolchainResolutionFunction}. */
@RunWith(JUnit4.class)
public class ToolchainResolutionFunctionTest extends ToolchainTestCase {

  private EvaluationResult<UnloadedToolchainContext> invokeToolchainResolution(SkyKey key)
      throws InterruptedException {
    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
  }

  @Test
  public void resolve() throws Exception {
    // This should select platform mac, toolchain extra_toolchain_mac, because platform
    // mac is listed first.
    addToolchain(
        "extra",
        "extra_toolchain_linux",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "baz");
    addToolchain(
        "extra",
        "extra_toolchain_mac",
        ImmutableList.of("//constraints:mac"),
        ImmutableList.of("//constraints:linux"),
        "baz");
    rewriteWorkspace(
        "register_toolchains('//extra:extra_toolchain_linux', '//extra:extra_toolchain_mac')",
        "register_execution_platforms('//platforms:mac', '//platforms:linux')");

    useConfiguration("--platforms=//platforms:linux");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey, ImmutableSet.of(testToolchainTypeLabel), ImmutableSet.of());

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext value = result.get(key);
    assertThat(value).isNotNull();

    assertThat(value.requiredToolchainTypes()).containsExactly(testToolchainType);
    assertThat(value.resolvedToolchainLabels())
        .containsExactly(Label.parseAbsoluteUnchecked("//extra:extra_toolchain_mac_impl"));

    assertThat(value.executionPlatform()).isNotNull();
    assertThat(value.executionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:mac"));

    assertThat(value.targetPlatform()).isNotNull();
    assertThat(value.targetPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));
  }

  @Test
  public void resolve_noToolchainType() throws Exception {
    scratch.file("host/BUILD", "platform(name = 'host')");
    rewriteWorkspace("register_execution_platforms('//platforms:mac', '//platforms:linux')");

    useConfiguration("--host_platform=//host:host", "--platforms=//platforms:linux");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(targetConfigKey, ImmutableSet.of(), ImmutableSet.of());

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext value = result.get(key);
    assertThat(value).isNotNull();

    assertThat(value.requiredToolchainTypes()).isEmpty();

    // With no toolchains requested, should fall back to the host platform.
    assertThat(value.executionPlatform()).isNotNull();
    assertThat(value.executionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//host:host"));

    assertThat(value.targetPlatform()).isNotNull();
    assertThat(value.targetPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));
  }

  @Test
  public void resolve_noToolchainType_hostNotAvailable() throws Exception {
    scratch.file("host/BUILD", "platform(name = 'host')");
    scratch.file(
        "sample/BUILD",
        "constraint_setting(name='demo')",
        "constraint_value(name = 'demo_a', constraint_setting=':demo')",
        "constraint_value(name = 'demo_b', constraint_setting=':demo')",
        "platform(name = 'sample_a',",
        "  constraint_values = [':demo_a'],",
        ")",
        "platform(name = 'sample_b',",
        "  constraint_values = [':demo_b'],",
        ")");
    rewriteWorkspace(
        "register_execution_platforms('//platforms:mac', '//platforms:linux',",
        "    '//sample:sample_a', '//sample:sample_b')");

    useConfiguration("--host_platform=//host:host", "--platforms=//platforms:linux");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey,
            ImmutableSet.of(),
            ImmutableSet.of(Label.parseAbsoluteUnchecked("//sample:demo_b")));

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext value = result.get(key);
    assertThat(value).isNotNull();

    assertThat(value.requiredToolchainTypes()).isEmpty();

    assertThat(value.executionPlatform()).isNotNull();
    assertThat(value.executionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//sample:sample_b"));

    assertThat(value.targetPlatform()).isNotNull();
    assertThat(value.targetPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));
  }

  @Test
  public void resolve_unavailableToolchainType_single() throws Exception {
    useConfiguration("--host_platform=//platforms:linux", "--platforms=//platforms:mac");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey,
            ImmutableSet.of(
                testToolchainTypeLabel, Label.parseAbsoluteUnchecked("//fake/toolchain:type_1")),
            ImmutableSet.of());

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(UnresolvedToolchainsException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("no matching toolchains found for types //fake/toolchain:type_1");
  }

  @Test
  public void resolve_unavailableToolchainType_multiple() throws Exception {
    useConfiguration("--host_platform=//platforms:linux", "--platforms=//platforms:mac");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey,
            ImmutableSet.of(
                testToolchainTypeLabel,
                Label.parseAbsoluteUnchecked("//fake/toolchain:type_1"),
                Label.parseAbsoluteUnchecked("//fake/toolchain:type_2")),
            ImmutableSet.of());

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(UnresolvedToolchainsException.class);
    // Only one of the missing types will be reported, so do not check the specific error message.
  }

  @Test
  public void resolve_invalidTargetPlatform_badTarget() throws Exception {
    scratch.file("invalid/BUILD", "filegroup(name = 'not_a_platform')");
    useConfiguration("--platforms=//invalid:not_a_platform");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey, ImmutableSet.of(testToolchainTypeLabel), ImmutableSet.of());

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(
            "//invalid:not_a_platform was referenced as a platform, "
                + "but does not provide PlatformInfo");
  }

  @Test
  public void resolve_invalidTargetPlatform_badPackage() throws Exception {
    scratch.resolve("invalid").delete();
    useConfiguration("--platforms=//invalid:not_a_platform");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey, ImmutableSet.of(testToolchainTypeLabel), ImmutableSet.of());

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("BUILD file not found");
  }

  @Test
  public void resolve_invalidHostPlatform() throws Exception {
    scratch.file("invalid/BUILD", "filegroup(name = 'not_a_platform')");
    useConfiguration("--host_platform=//invalid:not_a_platform");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey, ImmutableSet.of(testToolchainTypeLabel), ImmutableSet.of());

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//invalid:not_a_platform");
  }

  @Test
  public void resolve_invalidExecutionPlatform() throws Exception {
    scratch.file("invalid/BUILD", "filegroup(name = 'not_a_platform')");
    useConfiguration("--extra_execution_platforms=//invalid:not_a_platform");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey, ImmutableSet.of(testToolchainTypeLabel), ImmutableSet.of());

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//invalid:not_a_platform");
  }

  @Test
  public void resolve_execConstraints() throws Exception {
    // This should select platform linux, toolchain extra_toolchain_linux, due to extra constraints,
    // even though platform mac is registered first.
    addToolchain(
        /* packageName= */ "extra",
        /* toolchainName= */ "extra_toolchain_linux",
        /* execConstraints= */ ImmutableList.of("//constraints:linux"),
        /* targetConstraints= */ ImmutableList.of("//constraints:linux"),
        /* data= */ "baz");
    addToolchain(
        /* packageName= */ "extra",
        /* toolchainName= */ "extra_toolchain_mac",
        /* execConstraints= */ ImmutableList.of("//constraints:mac"),
        /* targetConstraints= */ ImmutableList.of("//constraints:linux"),
        /* data= */ "baz");
    rewriteWorkspace(
        "register_toolchains('//extra:extra_toolchain_linux', '//extra:extra_toolchain_mac')",
        "register_execution_platforms('//platforms:mac', '//platforms:linux')");

    useConfiguration("--platforms=//platforms:linux");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey,
            ImmutableSet.of(testToolchainTypeLabel),
            ImmutableSet.of(Label.parseAbsoluteUnchecked("//constraints:linux")));

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext value = result.get(key);
    assertThat(value).isNotNull();

    assertThat(value.requiredToolchainTypes()).containsExactly(testToolchainType);
    assertThat(value.resolvedToolchainLabels())
        .containsExactly(Label.parseAbsoluteUnchecked("//extra:extra_toolchain_linux_impl"));

    assertThat(value.executionPlatform()).isNotNull();
    assertThat(value.executionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));

    assertThat(value.targetPlatform()).isNotNull();
    assertThat(value.targetPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));
  }

  @Test
  public void resolve_execConstraints_invalid() throws Exception {
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey,
            ImmutableSet.of(testToolchainTypeLabel),
            ImmutableSet.of(Label.parseAbsoluteUnchecked("//platforms:linux")));

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidConstraintValueException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//platforms:linux");
  }

  @Test
  public void resolve_noMatchingPlatform() throws Exception {
    // Write toolchain A, and a toolchain implementing it.
    scratch.appendFile(
        "a/BUILD",
        "toolchain_type(name = 'toolchain_type_A')",
        "toolchain(",
        "    name = 'toolchain',",
        "    toolchain_type = ':toolchain_type_A',",
        "    exec_compatible_with = ['//constraints:mac'],",
        "    target_compatible_with = [],",
        "    toolchain = ':toolchain_impl')",
        "filegroup(name='toolchain_impl')");
    // Write toolchain B, and a toolchain implementing it.
    scratch.appendFile(
        "b/BUILD",
        "load('//toolchain:toolchain_def.bzl', 'test_toolchain')",
        "toolchain_type(name = 'toolchain_type_B')",
        "toolchain(",
        "    name = 'toolchain',",
        "    toolchain_type = ':toolchain_type_B',",
        "    exec_compatible_with = ['//constraints:linux'],",
        "    target_compatible_with = [],",
        "    toolchain = ':toolchain_impl')",
        "filegroup(name='toolchain_impl')");

    rewriteWorkspace(
        "register_toolchains('//a:toolchain', '//b:toolchain')",
        "register_execution_platforms('//platforms:mac', '//platforms:linux')");

    useConfiguration("--platforms=//platforms:linux");
    UnloadedToolchainContext.Key key =
        UnloadedToolchainContext.key(
            targetConfigKey,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//a:toolchain_type_A"),
                Label.parseAbsoluteUnchecked("//b:toolchain_type_B")),
            ImmutableSet.of());

    EvaluationResult<UnloadedToolchainContext> result = invokeToolchainResolution(key);
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(NoMatchingPlatformException.class);
  }
}
