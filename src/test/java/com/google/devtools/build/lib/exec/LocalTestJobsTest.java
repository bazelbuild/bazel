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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.exec.ExecutionOptions.LocalTestJobsConverter;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link com.google.devtools.build.lib.exec.ExecutionOptions.LocalTestJobsConverter}. */
@RunWith(JUnit4.class)
public class LocalTestJobsTest {

  private LocalTestJobsConverter localTestJobsConverter;

  @Before
  public void setUp() throws OptionsParsingException {
    localTestJobsConverter = new LocalTestJobsConverter();
  }

  @Test
  public void testLocalTestJobsMustBePositive() throws OptionsParsingException {
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> localTestJobsConverter.convert("-1"));
    assertThat(thrown).hasMessageThat().contains("must be at least 0");
  }

  @Test
  public void testLocalTestJobsAutoIsZero() throws OptionsParsingException {
    assertThat(localTestJobsConverter.convert("auto")).isEqualTo(0);
  }
}
