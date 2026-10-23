// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.analysis.test.TestTrimmingLogic;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.Options;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link DefaultPlatformConfigurationProvider} and {@link
 * SettablePlatformConfigurationProvider}.
 */
@RunWith(JUnit4.class)
public final class DefaultPlatformConfigurationProviderTest extends AnalysisTestCase {

  private DefaultPlatformConfigurationProvider provider;

  @Before
  public void setUpProvider() throws Exception {
    scratch.file("test/BUILD", "filegroup(name = 'test')");
    update("//test:test");
    BuildOptions targetOptions = getTargetConfiguration().getOptions();
    BuildOptions execOptions = getExecConfiguration().getOptions();
    Label topLevelPlatform = targetOptions.get(PlatformOptions.class).computeTargetPlatform();
    provider =
        new DefaultPlatformConfigurationProvider(topLevelPlatform, targetOptions, execOptions);
  }

  @Test
  public void testGetTopLevelPlatformLabel() throws Exception {
    Label label = provider.getTopLevelPlatformLabel();
    Label expected =
        getTargetConfiguration().getOptions().get(PlatformOptions.class).computeTargetPlatform();
    assertThat(label).isEqualTo(expected);
  }

  @Test
  public void testGetBaseOptionsForPlatform_targetAndExec() throws Exception {
    Label customPlatform = Label.parseCanonicalUnchecked("//custom:platform");

    BuildOptions targetBase =
        provider.getBaseOptionsForPlatform(
            customPlatform, /* isExec= */ false, /* trimTestOptions= */ false);
    assertThat(targetBase).isNotNull();
    assertThat(targetBase.get(PlatformOptions.class).getPlatforms())
        .containsExactly(customPlatform);

    BuildOptions execBase =
        provider.getBaseOptionsForPlatform(
            customPlatform, /* isExec= */ true, /* trimTestOptions= */ false);
    assertThat(execBase).isNotNull();
    assertThat(execBase.get(PlatformOptions.class).getPlatforms()).containsExactly(customPlatform);
  }

  @Test
  public void testGetBaseOptionsForPlatform_trimTestOptions() throws Exception {
    Label customPlatform = Label.parseCanonicalUnchecked("//custom:platform");

    BuildOptions trimmed =
        provider.getBaseOptionsForPlatform(
            customPlatform, /* isExec= */ false, /* trimTestOptions= */ true);
    assertThat(trimmed.contains(TestOptions.class)).isFalse();
  }

  @Test
  public void testGetBaseOptionsForPlatform_untrimmed_doesNotDependOnThisBuildsTestOptions()
      throws Exception {
    // Test options and the --run_under command are not covered by FrontierNodeVersion, so the
    // baseline that untrimmed keys are diffed against must not depend on them (b/570095568).
    Label platform = Label.parseCanonicalUnchecked("//custom:platform");
    BuildOptions withTestFlags =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class, TestOptions.class),
            "--platforms=//custom:platform",
            "--experimental_cancel_concurrent_tests",
            "--test_arg=foo",
            "--run_under=//tools:runner --arg");
    BuildOptions withoutTestFlags =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class, TestOptions.class),
            "--platforms=//custom:platform",
            "--run_under=//tools:runner");
    var providerWithTestFlags =
        new DefaultPlatformConfigurationProvider(platform, withTestFlags, withTestFlags);
    var providerWithoutTestFlags =
        new DefaultPlatformConfigurationProvider(platform, withoutTestFlags, withoutTestFlags);

    for (boolean isExec : new boolean[] {false, true}) {
      BuildOptions baselineWithTestFlags =
          providerWithTestFlags.getBaseOptionsForPlatform(
              platform, isExec, /* trimTestOptions= */ false);
      BuildOptions baselineWithoutTestFlags =
          providerWithoutTestFlags.getBaseOptionsForPlatform(
              platform, isExec, /* trimTestOptions= */ false);

      assertThat(baselineWithTestFlags).isEqualTo(baselineWithoutTestFlags);
      assertThat(baselineWithTestFlags.get(TestOptions.class))
          .isEqualTo(Options.getDefaults(TestOptions.class));
      assertThat(baselineWithTestFlags.get(CoreOptions.class).getRunUnder())
          .isEqualTo(TestTrimmingLogic.trim(withTestFlags).get(CoreOptions.class).getRunUnder());
    }
  }

  @Test
  public void testGetActualBaseOptionsForPlatform_keepsThisBuildsTestOptions() throws Exception {
    Label platform = Label.parseCanonicalUnchecked("//custom:platform");
    BuildOptions topLevelOptions =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class, TestOptions.class),
            "--platforms=//custom:platform",
            "--test_filter=Foo",
            "--run_under=//tools:runner --arg");
    var providerWithTestFlags =
        new DefaultPlatformConfigurationProvider(platform, topLevelOptions, topLevelOptions);

    for (boolean isExec : new boolean[] {false, true}) {
      BuildOptions untrimmed =
          providerWithTestFlags.getActualBaseOptionsForPlatform(
              platform, isExec, /* trimTestOptions= */ false);
      assertThat(untrimmed.get(TestOptions.class))
          .isEqualTo(topLevelOptions.get(TestOptions.class));
      assertThat(untrimmed.get(CoreOptions.class).getRunUnder())
          .isEqualTo(topLevelOptions.get(CoreOptions.class).getRunUnder());

      // Trimmed configurations have no test options, so both baselines are the same.
      assertThat(
              providerWithTestFlags.getActualBaseOptionsForPlatform(
                  platform, isExec, /* trimTestOptions= */ true))
          .isEqualTo(
              providerWithTestFlags.getBaseOptionsForPlatform(
                  platform, isExec, /* trimTestOptions= */ true));
    }
  }

  @Test
  public void testResolveMnemonic_withNonDefaultTestOptions_matchesConfigurationMnemonic()
      throws Exception {
    // The baseline used for serialization has default test options, but output directory naming
    // has to keep using the real baseline; otherwise the non-default test options would add an
    // ST-hash that Skyframe never added.
    scratch.file("custom/BUILD", "platform(name = 'my_platform')");
    useConfiguration(
        "--platforms=//custom:my_platform",
        "--experimental_cancel_concurrent_tests",
        "--test_arg=foo");
    update("//test:test");

    BuildOptions targetOptions = getTargetConfiguration().getOptions();
    assertThat(targetOptions.contains(TestOptions.class)).isTrue();
    BuildOptions execOptions = getExecConfiguration().getOptions();
    Label topLevelPlatform = targetOptions.get(PlatformOptions.class).computeTargetPlatform();
    provider =
        new DefaultPlatformConfigurationProvider(topLevelPlatform, targetOptions, execOptions);

    String mnemonic = provider.resolveMnemonic(targetOptions);
    assertThat(mnemonic).doesNotContain("-ST-");
    assertThat(mnemonic).isEqualTo(getTargetConfiguration().getMnemonic());
  }

  @Test
  public void testResolveMnemonic_withoutPlatformInOutputDir_throwsIllegalArgumentException()
      throws Exception {
    scratch.file("custom/BUILD", "platform(name = 'p1')", "platform(name = 'p2')");
    useConfiguration(
        "--platforms=//custom:p1", "--incompatible_limit_platforms_in_output_dir_to=//custom:p2");
    update("//test:test");

    BuildOptions targetOptions = getTargetConfiguration().getOptions();
    assertThrows(IllegalArgumentException.class, () -> provider.resolveMnemonic(targetOptions));
  }

  @Test
  public void testResolveMnemonic_nullOptions_throwsNullPointerException() {
    assertThrows(NullPointerException.class, () -> provider.resolveMnemonic(null));
  }

  @Test
  public void testResolveMnemonic_withPlatformInOutputDir() throws Exception {
    scratch.file("custom/BUILD", "platform(name = 'my_platform')");
    useConfiguration("--platforms=//custom:my_platform", "--compilation_mode=dbg");
    update("//test:test");

    BuildOptions targetOptions = getTargetConfiguration().getOptions();
    BuildOptions execOptions = getExecConfiguration().getOptions();
    Label topLevelPlatform = targetOptions.get(PlatformOptions.class).computeTargetPlatform();
    provider =
        new DefaultPlatformConfigurationProvider(topLevelPlatform, targetOptions, execOptions);

    String mnemonic = provider.resolveMnemonic(targetOptions);
    assertThat(mnemonic).isNotEmpty();
    String legacyMnemonic = targetOptions.get(CoreOptions.class).getCpu() + "-dbg";
    assertThat(mnemonic).isNotEqualTo(legacyMnemonic);
    assertThat(mnemonic).endsWith("-dbg");
    assertThat(mnemonic).startsWith("my_platform");
  }

  @Test
  public void testSettableProvider_uninitializedThrowsIllegalStateException() {
    var settableProvider = new SettablePlatformConfigurationProvider();
    assertThrows(IllegalStateException.class, settableProvider::getTopLevelPlatformLabel);
  }

  @Test
  public void testSettableProvider_setOnceWorksAndDelegates() {
    var settableProvider = new SettablePlatformConfigurationProvider();
    settableProvider.setOnce(provider);
    assertThat(settableProvider.getTopLevelPlatformLabel())
        .isEqualTo(provider.getTopLevelPlatformLabel());
  }

  @Test
  public void testSettableProvider_setTwiceThrowsIllegalStateException() {
    var settableProvider = new SettablePlatformConfigurationProvider();
    settableProvider.setOnce(provider);
    assertThrows(IllegalStateException.class, () -> settableProvider.setOnce(provider));
  }
}
