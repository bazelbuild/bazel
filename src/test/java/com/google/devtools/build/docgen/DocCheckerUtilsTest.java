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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

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
    assertNull(DocCheckerUtils.getFirstUnclosedTag("<html></html>"));
    assertEquals("ol", DocCheckerUtils.getFirstUnclosedTag("<html><ol></html>"));
    assertEquals("ol", DocCheckerUtils.getFirstUnclosedTag("<html><ol><li>foo</li></html>"));
    assertEquals("ol", DocCheckerUtils.getFirstUnclosedTag("<html><ol><li/>foo<li/>bar</html>"));
  }

  @Test
  public void testUncheckedTagsDontFire() {
    assertNull(DocCheckerUtils.getFirstUnclosedTag("<html><br></html>"));
    assertNull(DocCheckerUtils.getFirstUnclosedTag("<html><li></html>"));
    assertNull(DocCheckerUtils.getFirstUnclosedTag("<html><ul></html>"));
    assertNull(DocCheckerUtils.getFirstUnclosedTag("<html><p></html>"));
  }
}
