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
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
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
  public void testTrimTestOptions() throws Exception {
    BuildOptions withTestOptions =
        BuildOptions.of(ImmutableList.of(CoreOptions.class, TestOptions.class));
    BuildOptions withoutTestOptions = BuildOptions.of(ImmutableList.of(CoreOptions.class));

    assertThat(provider.trimTestOptions(withTestOptions)).isFalse();
    assertThat(provider.trimTestOptions(withoutTestOptions)).isTrue();
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
