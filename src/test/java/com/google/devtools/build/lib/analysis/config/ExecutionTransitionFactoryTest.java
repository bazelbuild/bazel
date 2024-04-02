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
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.testutil.FakeAttributeMapper;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import java.lang.reflect.Field;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ExecutionTransitionFactory}. */
@RunWith(TestParameterInjector.class)
public class ExecutionTransitionFactoryTest extends BuildViewTestCase {
  private static final Label EXECUTION_PLATFORM = Label.parseCanonicalUnchecked("//platform:exec");

  private PatchTransition getExecTransition(Label execPlatform) throws Exception {
    return ExecutionTransitionFactory.createFactory()
        .create(
            AttributeTransitionData.builder()
                .attributes(FakeAttributeMapper.empty())
                .analysisData(
                    getSkyframeExecutor()
                        .getStarlarkExecTransitionForTesting(targetConfig.getOptions(), reporter))
                .executionPlatform(execPlatform)
                .build());
  }

  @Test
  public void executionTransition() throws Exception {
    PatchTransition transition = getExecTransition(EXECUTION_PLATFORM);
    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            targetConfig.getOptions().getFragmentClasses(), "--platforms=//platform:target");

    BuildOptions result =
        transition.patch(
            new BuildOptionsView(options, transition.requiresOptionFragments()),
            new StoredEventHandler());
    assertThat(result).isNotNull();
    assertThat(result).isNotSameInstanceAs(options);

    assertThat(result.contains(CoreOptions.class)).isNotNull();
    assertThat(result.get(CoreOptions.class).isExec).isTrue();
    assertThat(result.contains(PlatformOptions.class)).isNotNull();
    assertThat(result.get(PlatformOptions.class).platforms).containsExactly(EXECUTION_PLATFORM);
  }

  @Test
  public void executionTransition_noExecPlatform() throws Exception {
    // No execution platform available.
    PatchTransition transition = getExecTransition(null);
    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            targetConfig.getOptions().getFragmentClasses(), "--platforms=//platform:target");

    BuildOptions result =
        transition.patch(
            new BuildOptionsView(options, transition.requiresOptionFragments()),
            new StoredEventHandler());
    assertThat(result).isNotNull();
    assertThat(result).isEqualTo(options);
  }

  @Test
  public void executionTransition_confDist_legacy() throws Exception {
    PatchTransition transition = getExecTransition(EXECUTION_PLATFORM);
    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            targetConfig.getOptions().getFragmentClasses(),
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
  public void executionTransition_confDist_fullHash() throws Exception {
    PatchTransition transition = getExecTransition(EXECUTION_PLATFORM);
    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            targetConfig.getOptions().getFragmentClasses(),
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
  public void executionTransition_confDist_diffToAffected() throws Exception {
    PatchTransition transition = getExecTransition(EXECUTION_PLATFORM);
    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            targetConfig.getOptions().getFragmentClasses(),
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
  public void executionTransition_confDist_off() throws Exception {
    PatchTransition transition = getExecTransition(EXECUTION_PLATFORM);
    assertThat(transition).isNotNull();

    // Apply the transition.
    BuildOptions options =
        BuildOptions.of(
            targetConfig.getOptions().getFragmentClasses(),
            "--platforms=//platform:target",
            "--experimental_exec_configuration_distinguisher=off");

    BuildOptions result =
        transition.patch(
            new BuildOptionsView(options, transition.requiresOptionFragments()),
            new StoredEventHandler());

    assertThat(result.get(CoreOptions.class).affectedByStarlarkTransition).isEmpty();
    assertThat(result.get(CoreOptions.class).platformSuffix).isEqualTo("exec");
  }

  @Test
  @TestParameters({
    "{cmdLineRef: 'gibberish', expectedError: 'Doesn''t match expected form"
        + " //pkg:file.bzl%%symbol'}",
    "{cmdLineRef: '//test:defs.bzl', expectedError: 'Doesn''t match expected form"
        + " //pkg:file.bzl%%symbol'}",
    "{cmdLineRef: '//test:defs.bzl%', expectedError: 'Doesn''t match expected form"
        + " //pkg:file.bzl%%symbol'}",
    "{cmdLineRef: '//test:defs.bzl%symbol_doesnt_exist', expectedError: 'symbol_doesnt_exist not"
        + " found in //test:defs.bzl'}",
    "{cmdLineRef: '//test:file_doesnt_exist.bzl%symbol', expectedError:"
        + " '''//test:file_doesnt_exist.bzl'': no such file'}",
    "{cmdLineRef: '//test:defs.bzl%not_a_transition', expectedError: 'not_a_transition is not a"
        + " Starlark transition.'}"
  })
  public void starlarkExecFlagBadReferences(String cmdLineRef, String expectedError)
      throws Exception {
    scratch.file("test/defs.bzl", "not_a_transition = 4");
    scratch.file("test/BUILD");

    InvalidConfigurationException e =
        assertThrows(
            InvalidConfigurationException.class,
            () -> useConfiguration("--experimental_exec_config=" + cmdLineRef));
    assertThat(e).hasMessageThat().contains(expectedError);
  }

  /** Checks all incompatible options propagate to the exec configuration. */
  @Test
  public void incompatibleOptionsPreservedInExec() throws Exception {
    BuildOptions defaultOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            targetConfig.getOptions().getFragmentClasses());
    ImmutableMap<String, OptionInfo> optionInfoMap = OptionInfo.buildMapFrom(defaultOptions);

    // Find all options with the INCOMPATIBLE_CHANGE metadata tag or start with "--incompatible_".
    ImmutableMap<String, OptionInfo> incompatibleOptions =
        optionInfoMap.entrySet().stream()
            .filter(
                o ->
                    o.getKey().startsWith("incompatible_")
                        || o.getValue().hasOptionMetadataTag(OptionMetadataTag.INCOMPATIBLE_CHANGE))
            .filter(
                o ->
                    o.getValue()
                        .getDefinition()
                        .getField()
                        .getType()
                        .isAssignableFrom(boolean.class))
            .filter(
                o -> !o.getValue().getDefinition().getField().isAnnotationPresent(Deprecated.class))
            // TODO: b/328442047 - Remove this when the flag is removed.
            .filter(
                // Skipping this explicitly because it is a no-op but can't be removed yet.
                o -> !o.getKey().equals("incompatible_enable_android_toolchain_resolution"))
            .collect(toImmutableMap(k -> k.getKey(), v -> v.getValue()));

    // Verify all "--incompatible_*" options also have the INCOMPATIBLE_CHANGE metadata tag.
    ImmutableList<String> missingMetadataTagOptions =
        incompatibleOptions.entrySet().stream()
            .filter(o -> !o.getValue().hasOptionMetadataTag(OptionMetadataTag.INCOMPATIBLE_CHANGE))
            .map(o -> "--" + o.getValue().getDefinition().getOptionName())
            .collect(toImmutableList());

    // Flip all incompatible (boolean) options to their non-default value.
    BuildOptions flipped = defaultOptions.clone(); // To be flipped by below logic.
    for (OptionInfo option : incompatibleOptions.values()) {
      Field field = option.getDefinition().getField();
      FragmentOptions fragment = flipped.get(option.getOptionClass());
      field.setBoolean(fragment, !field.getBoolean(fragment));
    }

    PatchTransition execTransition = getExecTransition(EXECUTION_PLATFORM);
    BuildOptions execOptions =
        execTransition.patch(
            new BuildOptionsView(flipped, execTransition.requiresOptionFragments()),
            new StoredEventHandler());

    // Find which incompatible options are different in the exec config (shouldn't be any).
    ImmutableMultimap.Builder<Class<? extends FragmentOptions>, OptionDefinition>
        unpreservedOptions = new ImmutableMultimap.Builder<>();
    for (OptionInfo incompatibleOption : incompatibleOptions.values()) {
      Field field = incompatibleOption.getDefinition().getField();
      Class<? extends FragmentOptions> optionClass = incompatibleOption.getOptionClass();
      if (field.getBoolean(execOptions.get(optionClass))
          != field.getBoolean(flipped.get(optionClass))) {
        unpreservedOptions.put(
            incompatibleOption.getOptionClass(), incompatibleOption.getDefinition());
      }
    }

    assertThat(missingMetadataTagOptions).isEmpty();
    assertThat(unpreservedOptions.build()).isEmpty();
  }
}
