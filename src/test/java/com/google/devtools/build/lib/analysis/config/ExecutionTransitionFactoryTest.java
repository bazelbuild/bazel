package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.testutil.FakeAttributeMapper;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExecutionTransitionFactory}. */
@RunWith(JUnit4.class)
public class ExecutionTransitionFactoryTest {
  private static final Label EXECUTION_PLATFORM = Label.parseAbsoluteUnchecked("//platform:exec");

  @Test
  public void executionTransition() throws OptionsParsingException {
    ExecutionTransitionFactory execTransitionFactory = new ExecutionTransitionFactory();
    PatchTransition transition =
        execTransitionFactory.create(
            AttributeTransitionData.create(FakeAttributeMapper.empty(), EXECUTION_PLATFORM));

    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:target");

    BuildOptions result = transition.patch(options);
    assertThat(result).isNotNull();
    assertThat(result).isNotSameInstanceAs(options);

    assertThat(result.contains(CoreOptions.class)).isNotNull();
    assertThat(result.get(CoreOptions.class).isHost).isFalse();
    assertThat(result.contains(PlatformOptions.class)).isNotNull();
    assertThat(result.get(PlatformOptions.class).platforms).containsExactly(EXECUTION_PLATFORM);
  }

  @Test
  public void executionTransition_noExecPlatform() throws OptionsParsingException {
    ExecutionTransitionFactory execTransitionFactory = new ExecutionTransitionFactory();
    // No execution platform available.
    PatchTransition transition =
        execTransitionFactory.create(
            AttributeTransitionData.create(FakeAttributeMapper.empty(), null));

    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:target");

    BuildOptions result = transition.patch(options);
    assertThat(result).isNotNull();
    assertThat(result).isEqualTo(options);
  }
}
