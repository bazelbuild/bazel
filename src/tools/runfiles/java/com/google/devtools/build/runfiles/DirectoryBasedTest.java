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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link DirectoryBased} implementation of {@link Runfiles}. */
@RunWith(JUnit4.class)
public final class DirectoryBasedTest {

  @Test
  public void testRlocation() throws Exception {
    // The DirectoryBased implementation simply joins the runfiles directory and the runfile's path
    // on a "/". DirectoryBased does not perform any normalization, nor does it check that the path
    // exists.
    DirectoryBased r = new DirectoryBased("foo/bar baz//qux/");
    assertThat(r.rlocation("arg")).isEqualTo("foo/bar baz//qux//arg");
  }

  @Test
  public void testCtorArgumentValidation() throws Exception {
    try {
      new DirectoryBased(null);
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    try {
      new DirectoryBased("");
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }

    new DirectoryBased("non-empty value is fine");
  }
}
