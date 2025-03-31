// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Streams.stream;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.testutil.FakeAttributeMapper;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExecutionTransitionFactory}. */
@RunWith(JUnit4.class)
public class ExecutionTransitionFactoryTest extends BuildViewTestCase {
  private static final Label EXECUTION_PLATFORM = Label.parseCanonicalUnchecked("//platform:exec");

  @Test
  public void executionTransition() throws OptionsParsingException, InterruptedException {
    PatchTransition transition =
        ExecutionTransitionFactory.createFactory()
            .create(
                AttributeTransitionData.builder()
                    .attributes(FakeAttributeMapper.empty())
                    .executionPlatform(EXECUTION_PLATFORM)
                    .build());

    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:target");

    BuildOptions result =
        transition.patch(
            new BuildOptionsView(options, transition.requiresOptionFragments()),
            new StoredEventHandler());
    assertThat(result).isNotNull();
    assertThat(result).isNotSameInstanceAs(options);

    assertThat(result.contains(CoreOptions.class));
    assertThat(result.get(CoreOptions.class).isExec).isTrue();
    assertThat(result.contains(PlatformOptions.class));
    assertThat(result.get(PlatformOptions.class).platforms).containsExactly(EXECUTION_PLATFORM);
  }

  @Test
  public void executionTransition_noExecPlatform()
      throws OptionsParsingException, InterruptedException {
    // No execution platform available.
    PatchTransition transition =
        ExecutionTransitionFactory.createFactory()
            .create(
                AttributeTransitionData.builder()
                    .attributes(FakeAttributeMapper.empty())
                    .executionPlatform(null)
                    .build());

    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:target");

    BuildOptions result =
        transition.patch(
            new BuildOptionsView(options, transition.requiresOptionFragments()),
            new StoredEventHandler());
    assertThat(result).isNotNull();
    assertThat(result).isEqualTo(options);
  }

  @Test
  public void executionTransition_confDist_legacy()
      throws OptionsParsingException, InterruptedException {
    PatchTransition transition =
        ExecutionTransitionFactory.createFactory()
            .create(
                AttributeTransitionData.builder()
                    .attributes(FakeAttributeMapper.empty())
                    .executionPlatform(EXECUTION_PLATFORM)
                    .build());

    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:target",
            "--experimental_exec_configuration_distinguisher=legacy");

    BuildOptions result =
        transition.patch(
            new BuildOptionsView(options, transition.requiresOptionFragments()),
            new StoredEventHandler());

    assertThat(result.get(CoreOptions.class).affectedByStarlarkTransition).isEmpty();
    assertThat(result.get(CoreOptions.class).platformSuffix)
        .contains(String.format("%X", EXECUTION_PLATFORM.getCanonicalForm().hashCode()));
  }

  @Test
  public void executionTransition_confDist_fullHash()
      throws OptionsParsingException, InterruptedException {
    PatchTransition transition =
        ExecutionTransitionFactory.createFactory()
            .create(
                AttributeTransitionData.builder()
                    .attributes(FakeAttributeMapper.empty())
                    .executionPlatform(EXECUTION_PLATFORM)
                    .build());

    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:target",
            "--experimental_exec_configuration_distinguisher=full_hash");

    BuildOptions result =
        transition.patch(
            new BuildOptionsView(options, transition.requiresOptionFragments()),
            new StoredEventHandler());

    BuildOptions mutableCopy = result.clone();
    mutableCopy.get(CoreOptions.class).platformSuffix = "";
    int fullHash = mutableCopy.hashCode();

    assertThat(result.get(CoreOptions.class).affectedByStarlarkTransition).isEmpty();
    assertThat(result.get(CoreOptions.class).platformSuffix)
        .contains(String.format("%X", fullHash));
  }

  @Test
  public void executionTransition_confDist_diffToAffected()
      throws OptionsParsingException, InterruptedException {
    PatchTransition transition =
        ExecutionTransitionFactory.createFactory()
            .create(
                AttributeTransitionData.builder()
                    .attributes(FakeAttributeMapper.empty())
                    .executionPlatform(EXECUTION_PLATFORM)
                    .build());

    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:target",
            "--experimental_exec_configuration_distinguisher=diff_to_affected");

    BuildOptions result =
        transition.patch(
            new BuildOptionsView(options, transition.requiresOptionFragments()),
            new StoredEventHandler());

    assertThat(result.get(CoreOptions.class).affectedByStarlarkTransition).isNotEmpty();
    assertThat(result.get(CoreOptions.class).platformSuffix).isEqualTo("exec");
  }

  @Test
  public void executionTransition_confDist_off()
      throws OptionsParsingException, InterruptedException {
    PatchTransition transition =
        ExecutionTransitionFactory.createFactory()
            .create(
                AttributeTransitionData.builder()
                    .attributes(FakeAttributeMapper.empty())
                    .executionPlatform(EXECUTION_PLATFORM)
                    .build());

    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:target",
            "--experimental_exec_configuration_distinguisher=off");

    BuildOptions result =
        transition.patch(
            new BuildOptionsView(options, transition.requiresOptionFragments()),
            new StoredEventHandler());

    assertThat(result.get(CoreOptions.class).affectedByStarlarkTransition).isEmpty();
    assertThat(result.get(CoreOptions.class).platformSuffix).isEqualTo("exec");
  }

  /**
   * Migration test for b/292619013.
   *
   * <p>The exec transition is moving to Starlark. The Starlark version is currently checked into
   * Blaze builtins and enabled by {@code --experimental_exec_config}.
   *
   * <p>That means both the native and Starlark versions co-exist until we're ready to use the
   * Starlark version exclusively and delete the native version. During this migration period we
   * must ensure they stay in sync. That's what this test checks.
   *
   * <p>Specifically, this test sets {@code --experimental_exec_config_diff}. That makes builds run
   * both the native and Starlark logic on any exec transition, compare their output, and print
   * differences as an INFO event. This test checks that the event message shows no differences.
   *
   * <p>If you see a difference, that means the Starlark transition is setting a flag value
   * differently than the native transition. The fix is to update one or both transitions to ensure
   * they're setting the flag the same way. Test error output should show which values differ.
   */
  // TODO(b/301644122): delete the native exec transition and this test.
  @Test
  public void testStarlarkExecTransitionMatchesNativeExecTransition() throws Exception {
    if (TestConstants.PRODUCT_NAME.equals("bazel")) {
      // TODO(b/301643153): check a Bazel-compatible Starlark transition into Bazel builtins.
      return;
    }
    scratch.file(
        "test/defs.bzl",
        "with_exec_transition = rule(",
        "  implementation = lambda ctx: [],",
        "  attrs = {",
        "    'dep': attr.label(cfg = 'exec'),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:defs.bzl', 'with_exec_transition')",
        "with_exec_transition(name = 'parent', dep = ':exec_configured_dep')",
        "with_exec_transition(name = 'exec_configured_dep')");
    useConfiguration(
        "--experimental_exec_config=@_builtins//:blaze/common/google_exec_platforms.bzl%google_exec_transition",
        "--experimental_exec_config_diff",
        // This flag's default value is {'Proguard': null}. null (the Java object) isn't readable
        // by Starlark transitions and crashes Blaze. This isn't a problem in production because
        // a global blazerc overrides the default. Do similar here. Also see b/294914034#comment3.
        "--experimental_bytecode_optimizers=Optimizer=//java/com/google/optimizationtest:optimizer");

    getConfiguredTarget("//test:parent");

    ImmutableList<Event> comparingTransitionEvents =
        stream(eventCollector.filtered(EventKind.INFO))
            .filter(e -> e.getMessage().contains("ComparingTransition"))
            .collect(toImmutableList());
    String comparingTransitionOutput =
        Iterables.getOnlyElement(comparingTransitionEvents).getMessage();

    assertThat(comparingTransitionOutput).contains("- unique fragments in starlark mode: none");
    assertThat(comparingTransitionOutput).contains("- unique fragments in native mode: none");
    assertThat(comparingTransitionOutput).contains("- total option differences: 0");
  }
}
