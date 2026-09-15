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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for DocCheckerUtils.
 */
@RunWith(JUnit4.class)
public class DocCheckerUtilsTest {

  @Test
  public void testUnclosedTags() {
    assertThat(DocCheckerUtils.getFirstUnclosedTag("<html></html>")).isNull();
    assertThat(DocCheckerUtils.getFirstUnclosedTag("<html><ol></html>")).isEqualTo("ol");
    assertThat(DocCheckerUtils.getFirstUnclosedTag("<html><ol><li>foo</li></html>"))
        .isEqualTo("ol");
    assertThat(DocCheckerUtils.getFirstUnclosedTag("<html><ol><li/>foo<li/>bar</html>"))
        .isEqualTo("ol");
  }

  @Test
  public void testUncheckedTagsDontFire() {
    assertThat(DocCheckerUtils.getFirstUnclosedTag("<html><br></html>")).isNull();
    assertThat(DocCheckerUtils.getFirstUnclosedTag("<html><li></html>")).isNull();
    assertThat(DocCheckerUtils.getFirstUnclosedTag("<html><ul></html>")).isNull();
    assertThat(DocCheckerUtils.getFirstUnclosedTag("<html><p></html>")).isNull();
  }
}
