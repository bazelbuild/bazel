// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for @{link RepositoryName}.
 */
@RunWith(JUnit4.class)
public class RepositoryNameTest {

  public void assertNotValid(String name, String expectedMessage) {
    try {
      RepositoryName.create(name);
      fail();
    } catch (LabelSyntaxException expected) {
      assertThat(expected.getMessage()).contains(expectedMessage);
    }
  }

  @Test
  public void testValidateRepositoryName() throws Exception {
    assertEquals("@foo", RepositoryName.create("@foo").toString());
    assertThat(RepositoryName.create("").toString()).isEmpty();
    assertEquals("@foo_bar", RepositoryName.create("@foo_bar").toString());
    assertEquals("@foo-bar", RepositoryName.create("@foo-bar").toString());
    assertEquals("@foo.bar", RepositoryName.create("@foo.bar").toString());
    assertEquals("@..foo", RepositoryName.create("@..foo").toString());
    assertEquals("@foo..", RepositoryName.create("@foo..").toString());
    assertEquals("@.foo", RepositoryName.create("@.foo").toString());

    assertNotValid("x", "workspace names must start with '@'");
    assertNotValid("@.", "workspace names are not allowed to be '@.'");
    assertNotValid("@..", "workspace names are not allowed to be '@..'");
    assertNotValid("@foo/bar", "workspace names may contain only A-Z, a-z, 0-9, '-', '_' and '.'");
    assertNotValid("@foo@", "workspace names may contain only A-Z, a-z, 0-9, '-', '_' and '.'");
    assertNotValid("@foo\0", "workspace names may contain only A-Z, a-z, 0-9, '-', '_' and '.'");
  }

}
