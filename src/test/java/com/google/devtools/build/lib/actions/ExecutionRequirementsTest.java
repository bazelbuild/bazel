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

package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link ExecutionRequirements}
 */
@RunWith(JUnit4.class)
public class ExecutionRequirementsTest {
  @Test
  public void testTaggedRequirementsCPU() throws Exception {
    int parsed;

    parsed = ExecutionRequirements.TAGGED_CPU.parse("1");
    assertThat(parsed).isEqualTo(1);

    try {
      ExecutionRequirements.TAGGED_CPU.parse("+04");
      fail("Expected a ValidationException");
    } catch (ExecutionRequirements.TaggedRequirement.ValidationException e) {
      assertThat(e).hasMessageThat().contains("must be in canonical format");
    }
  }

  @Test
  public void testTaggedRequirementsMemory() throws Exception {
    int parsed;

    parsed = ExecutionRequirements.TAGGED_MEM_MB.parse("1536M");
    assertThat(parsed).isEqualTo(1536);

    parsed = ExecutionRequirements.TAGGED_MEM_MB.parse("1G");
    assertThat(parsed).isEqualTo(1024);

    parsed = ExecutionRequirements.TAGGED_MEM_MB.parse("1.5G");
    assertThat(parsed).isEqualTo(1536);

    try {
      ExecutionRequirements.TAGGED_MEM_MB.parse("1024");
      fail("Expected a ValidationException");
    } catch (ExecutionRequirements.TaggedRequirement.ValidationException e) {
      assertThat(e).hasMessageThat().contains("must match pattern");
    }
  }
}
