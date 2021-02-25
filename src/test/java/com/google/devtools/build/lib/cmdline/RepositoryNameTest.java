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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for @{link RepositoryName}.
 */
@RunWith(JUnit4.class)
public class RepositoryNameTest {

  public void assertNotValid(String name, String expectedMessage) {
    LabelSyntaxException expected =
        assertThrows(LabelSyntaxException.class, () -> RepositoryName.create(name));
    assertThat(expected).hasMessageThat().contains(expectedMessage);
  }

  @Test
  public void testValidateRepositoryName() throws Exception {
    assertThat(RepositoryName.create("@foo").toString()).isEqualTo("@foo");
    assertThat(RepositoryName.create("").toString()).isEmpty();
    assertThat(RepositoryName.create("@foo_bar").toString()).isEqualTo("@foo_bar");
    assertThat(RepositoryName.create("@foo-bar").toString()).isEqualTo("@foo-bar");
    assertThat(RepositoryName.create("@foo.bar").toString()).isEqualTo("@foo.bar");
    assertThat(RepositoryName.create("@..foo").toString()).isEqualTo("@..foo");
    assertThat(RepositoryName.create("@foo..").toString()).isEqualTo("@foo..");
    assertThat(RepositoryName.create("@.foo").toString()).isEqualTo("@.foo");

    assertNotValid("x", "workspace names must start with '@'");
    assertNotValid("@.", "workspace names are not allowed to be '@.'");
    assertNotValid("@..", "workspace names are not allowed to be '@..'");
    assertNotValid("@foo/bar", "workspace names may contain only A-Z, a-z, 0-9, '-', '_' and '.'");
    assertNotValid("@foo@", "workspace names may contain only A-Z, a-z, 0-9, '-', '_' and '.'");
    assertNotValid("@foo\0", "workspace names may contain only A-Z, a-z, 0-9, '-', '_' and '.'");
  }

  @Test
  public void testRunfilesDir() throws Exception {
    assertThat(RepositoryName.create("@foo").getRunfilesPath())
        .isEqualTo(PathFragment.create("../foo"));
    assertThat(RepositoryName.create("@").getRunfilesPath())
        .isEqualTo(PathFragment.EMPTY_FRAGMENT);
    assertThat(RepositoryName.create("").getRunfilesPath())
        .isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void testGetDefaultCanonicalForm() throws Exception {
    assertThat(RepositoryName.create("").getCanonicalForm()).isEqualTo("");
    assertThat(RepositoryName.create("@").getCanonicalForm()).isEqualTo("");
    assertThat(RepositoryName.create("@foo").getCanonicalForm()).isEqualTo("@foo");
  }
}
