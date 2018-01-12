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

package com.google.devtools.build.runfiles;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ManifestBased} implementation of {@link Runfiles}. */
@RunWith(JUnit4.class)
public final class ManifestBasedTest {

  @Test
  public void testCtorArgumentValidation() throws Exception {
    try {
      new ManifestBased(null);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    try {
      new ManifestBased("");
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    try (MockFile mf = new MockFile(ImmutableList.of("a b"))) {
      new ManifestBased(mf.path.toString());
    }
  }

  @Test
  public void testRlocation() throws Exception {
    try (MockFile mf =
        new MockFile(
            ImmutableList.of(
                "Foo/runfile1 C:/Actual Path\\runfile1",
                "Foo/Bar/runfile2 D:\\the path\\run file 2.txt"))) {
      ManifestBased r = new ManifestBased(mf.path.toString());
      assertThat(r.rlocation("Foo/runfile1")).isEqualTo("C:/Actual Path\\runfile1");
      assertThat(r.rlocation("Foo/Bar/runfile2")).isEqualTo("D:\\the path\\run file 2.txt");
      assertThat(r.rlocation("unknown")).isNull();
    }
  }
}
