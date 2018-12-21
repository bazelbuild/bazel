// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption.LoadingPhaseThreadCountConverter;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link LoadingPhaseThreadCountConverter}. */
@RunWith(JUnit4.class)
public class LoadingPhaseThreadsConverterTest {

  private LoadingPhaseThreadCountConverter loadingPhaseThreadCountConverter;

  @Before
  public void setUp() throws OptionsParsingException {
    loadingPhaseThreadCountConverter = new LoadingPhaseThreadCountConverter();
  }

  @Test
  public void testAutoLoadingPhaseThreadsUsesHardwareSettings() throws Exception {
    LocalHostCapacity.setLocalHostCapacity(ResourceSet.createWithRamCpu(0, 7));
    assertThat(loadingPhaseThreadCountConverter.convert("auto")).isEqualTo(7);
  }

  @Test
  public void testAutoLoadingPhaseThreadsCappedForTests() throws Exception {
    LocalHostCapacity.setLocalHostCapacity(ResourceSet.createWithRamCpu(0, 123));
    assertThat(loadingPhaseThreadCountConverter.convert("auto")).isEqualTo(20);
  }

  @Test
  public void testExplicitLoadingPhaseThreadsCappedForTests() throws Exception {
    assertThat(loadingPhaseThreadCountConverter.convert("200")).isEqualTo(20);
  }

  @Test
  public void testExplicitLoadingPhaseThreadsMustBeAtLeast1() throws Exception {
    OptionsParsingException thrown =
        assertThrows(
            OptionsParsingException.class, () -> loadingPhaseThreadCountConverter.convert("0"));
    assertThat(thrown).hasMessageThat().contains("must be at least 1");
  }
}
