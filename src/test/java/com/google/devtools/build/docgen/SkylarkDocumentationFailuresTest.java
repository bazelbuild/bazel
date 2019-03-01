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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for various failure modes of Skylark documentation generation. These are separate from
 * other documentation generation tests because the generator does not tolerate broken modules
 * anywhere in the classpath.
 */
@RunWith(JUnit4.class)
public class SkylarkDocumentationFailuresTest extends SkylarkTestCase {

  @Before
  public final void createBuildFile() throws Exception {
    scratch.file("foo/BUILD",
        "genrule(name = 'foo',",
        "  cmd = 'dummy_cmd',",
        "  srcs = ['a.txt', 'b.img'],",
        "  tools = ['t.exe'],",
        "  outs = ['c.txt'])");
  }

  @Override
  protected EvaluationTestCase createEvaluationTestCase(StarlarkSemantics semantics) {
    return new EvaluationTestCase();
  }

  /** MockClassCommonNameOne */
  @SkylarkModule(name = "MockClassCommonName",
      doc = "MockClassCommonName")
  private static class MockClassCommonNameOne {

    @SkylarkCallable(name = "one", doc = "one")
    public Integer one() {
      return 1;
    }
  }

  /** MockClassCommonNameTwo */
  @SkylarkModule(name = "MockClassCommonName",
      doc = "MockClassCommonName")
  private static class MockClassCommonNameTwo {

    @SkylarkCallable(name = "two", doc = "two")
    public Integer two() {
      return 1;
    }
  }

  /** PointsToCommonName */
  @SkylarkModule(name = "PointsToCommonName",
      doc = "PointsToCommonName")
  private static class PointsToCommonName {
    @SkylarkCallable(name = "one", doc = "one")
    public MockClassCommonNameOne getOne() {
      return null;
    }

    @SkylarkCallable(name = "two", doc = "two")
    public MockClassCommonNameTwo getTwo() {
      return null;
    }
  }

  @Test
  public void testModuleNameConflict() {
    IllegalStateException expected =
        assertThrows(IllegalStateException.class, () -> collect(PointsToCommonName.class));
    assertThat(expected.getMessage()).contains("are both modules with the same documentation");
  }

  private Map<String, SkylarkModuleDoc> collect(Class<?> classObject) {
    return SkylarkDocumentationCollector.collectModules(ImmutableList.of(classObject));
  }
}
