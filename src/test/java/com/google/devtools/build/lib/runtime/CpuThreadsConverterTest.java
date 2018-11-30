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

import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link CpuThreadsConverter}. */
@RunWith(JUnit4.class)
public class CpuThreadsConverterTest {

  public CpuThreadsConverter cpuThreadsConverter;

  @Before
  public void setUp() throws OptionsParsingException {
    cpuThreadsConverter = new CpuThreadsConverter();
  }

  @Test
  public void testCpuThreadsConverterCapsForTests() throws Exception {
    assertThat(cpuThreadsConverter.convert("200")).isEqualTo(20);
  }

  @Test
  public void testCpuThreadsConverterFailsOnThreadsLessThan1() throws Exception {
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> cpuThreadsConverter.adjustValue(0));
    assertThat(thrown).hasMessageThat().contains("must be at least 1");
  }
}
