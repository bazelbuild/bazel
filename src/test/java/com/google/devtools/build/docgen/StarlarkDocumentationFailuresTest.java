// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for various failure modes of Starlark documentation generation. These are separate from
 * other documentation generation tests because the generator does not tolerate broken modules
 * anywhere in the classpath.
 */
@RunWith(JUnit4.class)
public final class StarlarkDocumentationFailuresTest {

  /** MockClassCommonNameOne */
  @StarlarkBuiltin(name = "MockClassCommonName", doc = "MockClassCommonName")
  private static class MockClassCommonNameOne implements StarlarkValue {

    @StarlarkMethod(name = "one", doc = "one")
    public Integer one() {
      return 1;
    }
  }

  /** MockClassCommonNameTwo */
  @StarlarkBuiltin(name = "MockClassCommonName", doc = "MockClassCommonName")
  private static class MockClassCommonNameTwo implements StarlarkValue {

    @StarlarkMethod(name = "two", doc = "two")
    public Integer two() {
      return 1;
    }
  }

  /** PointsToCommonName */
  @StarlarkBuiltin(name = "PointsToCommonName", doc = "PointsToCommonName")
  private static class PointsToCommonName implements StarlarkValue {
    @StarlarkMethod(name = "one", doc = "one")
    public MockClassCommonNameOne getOne() {
      return null;
    }

    @StarlarkMethod(name = "two", doc = "two")
    public MockClassCommonNameTwo getTwo() {
      return null;
    }
  }

  @Test
  public void testModuleNameConflict() {
    IllegalStateException ex =
        assertThrows(
            IllegalStateException.class,
            () ->
                StarlarkDocumentationCollector.collectModules(
                    ImmutableList.of(PointsToCommonName.class)));
    assertThat(ex).hasMessageThat().contains("are both modules with the same documentation");
  }
}
