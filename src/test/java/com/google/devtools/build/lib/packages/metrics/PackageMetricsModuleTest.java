// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.metrics;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.OptionsParser;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PackageMetricsModule}. */
@RunWith(JUnit4.class)
public class PackageMetricsModuleTest {

  private final PackageMetricsPackageLoadingListener listener =
      PackageMetricsPackageLoadingListener.getInstance();

  private PackageMetricsModule underTest;

  @Before
  public void setUp() {
    underTest = new PackageMetricsModule();
    listener.setPackageMetricsRecorder(null);
  }

  @Test
  public void testBeforeCommandConfiguresNumberOfPackagesToTrack() throws Exception {
    underTest.beforeCommand(commandEnv("--log_top_n_packages=100"));
    assertThat(listener.getPackageMetricsRecorder())
        .isInstanceOf(ExtremaPackageMetricsRecorder.class);
    ExtremaPackageMetricsRecorder ext =
        (ExtremaPackageMetricsRecorder) listener.getPackageMetricsRecorder();
    assertThat(ext.getNumPackagesToTrack()).isEqualTo(100);
  }

  @Test
  public void testBeforeCommandConfiguresNumberOfPackagesToTrackTreatsNegativeAsZero()
      throws Exception {
    underTest.beforeCommand(commandEnv("--log_top_n_packages=-100"));
    assertThat(listener.getPackageMetricsRecorder())
        .isInstanceOf(ExtremaPackageMetricsRecorder.class);
    ExtremaPackageMetricsRecorder ext =
        (ExtremaPackageMetricsRecorder) listener.getPackageMetricsRecorder();
    assertThat(ext.getNumPackagesToTrack()).isEqualTo(0);
  }

  @Test
  public void testBeforeCommandConfiguresNumberOfPackagesToTrackButRequestsComplete()
      throws Exception {
    underTest.beforeCommand(
        commandEnv("--log_top_n_packages=100", "--record_metrics_for_all_packages=1"));
    assertThat(listener.getPackageMetricsRecorder())
        .isInstanceOf(CompletePackageMetricsRecorder.class);
  }

  @Test
  public void testAfterCommandGetsAndResetsMetrics() {
    // Mocking here is lazy, but it helps verify we actually did something with all of the results.
    PackageMetricsRecorder mockRecorder = mock(PackageMetricsRecorder.class);
    listener.setPackageMetricsRecorder(mockRecorder);

    underTest.afterCommand();
    verify(mockRecorder).loadingFinished();
  }

  private static CommandEnvironment commandEnv(String... options) throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(PackageMetricsModule.Options.class).build();
    parser.parse(options);

    CommandEnvironment mockEnv = mock(CommandEnvironment.class);
    when(mockEnv.getOptions()).thenReturn(parser);
    return mockEnv;
  }
}
