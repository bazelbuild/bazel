// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.runfiles;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link AutoBazelRepository}.
 */
@RunWith(JUnit4.class)
@AutoBazelRepository
public final class AutoBazelRepositoryTest {

  @AutoBazelRepository
  @SuppressWarnings("unused")
  private static class NestedClass {

    @AutoBazelRepository
    @SuppressWarnings("unused")
    private static class FurtherNestedClass {

    }
  }

  @Test
  public void testAutoBazelRepositoryValue() {
    @AutoBazelRepository
    @SuppressWarnings("unused")
    class NestedClass {

    }

    assertThat(AutoBazelRepository_AutoBazelRepositoryTest.NAME).isEqualTo(
        "for testing only");
    assertThat(AutoBazelRepository_AutoBazelRepositoryTest_NestedClass.NAME).isEqualTo(
        "for testing only");
    assertThat(
        AutoBazelRepository_AutoBazelRepositoryTest_NestedClass_FurtherNestedClass.NAME)
        .isEqualTo("for testing only");
  }
}
