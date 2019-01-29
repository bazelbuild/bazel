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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions.JobsConverter;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link com.google.devtools.build.lib.buildtool.BuildRequestOptions.JobsConverter}. */
@RunWith(JUnit4.class)
public class JobsConverterTest {

  JobsConverter jobsConverter;

  @Before
  public void setUp() {
    jobsConverter = new JobsConverter();
  }

  @Test
  public void testAutoJobsUsesHardwareSettings() throws Exception {
    LocalHostCapacity.setLocalHostCapacity(ResourceSet.createWithRamCpu(0, 123));
    assertThat(jobsConverter.convert("auto")).isEqualTo(123);
  }

  @Test
  public void testAutoJobsAdjustsIfHardwareDetectionIsBogus() throws Exception {
    LocalHostCapacity.setLocalHostCapacity(
        ResourceSet.createWithRamCpu(0, BuildRequestOptions.MAX_JOBS + 1));
    assertThat(jobsConverter.convert("auto")).isEqualTo(BuildRequestOptions.MAX_JOBS);
  }

  @Test
  public void testExplicitJobsLimited() throws Exception {
    assertThat(jobsConverter.convert(Integer.toString(BuildRequestOptions.MAX_JOBS + 1)))
        .isEqualTo(BuildRequestOptions.MAX_JOBS);
  }

  @Test
  public void testUnboundedJobsDeprecated() {
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> jobsConverter.convert("-1"));
    assertThat(thrown).hasMessageThat().contains("must be at least 1");
  }
}
