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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.PathFragment;
import net.starlark.java.eval.EvalException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RepositoryName}. */
@RunWith(JUnit4.class)
public class RepositoryNameTest {

  public void assertNotValid(String name, String expectedMessage) {
    LabelSyntaxException expected =
        assertThrows(LabelSyntaxException.class, () -> RepositoryName.create(name));
    assertThat(expected).hasMessageThat().contains(expectedMessage);
  }

  @Test
  public void testValidateRepositoryName() throws Exception {
    assertThat(RepositoryName.create("foo").getNameWithAt()).isEqualTo("@foo");
    assertThat(RepositoryName.create("").getNameWithAt()).isEqualTo("@");
    assertThat(RepositoryName.create("")).isSameInstanceAs(RepositoryName.MAIN);
    assertThat(RepositoryName.create("foo_bar").getNameWithAt()).isEqualTo("@foo_bar");
    assertThat(RepositoryName.create("foo-bar").getNameWithAt()).isEqualTo("@foo-bar");
    assertThat(RepositoryName.create("foo.bar").getNameWithAt()).isEqualTo("@foo.bar");
    assertThat(RepositoryName.create("..foo").getNameWithAt()).isEqualTo("@..foo");
    assertThat(RepositoryName.create("foo..").getNameWithAt()).isEqualTo("@foo..");
    assertThat(RepositoryName.create(".foo").getNameWithAt()).isEqualTo("@.foo");
    assertThat(RepositoryName.create("foo~bar").getNameWithAt()).isEqualTo("@foo~bar");

    assertNotValid(".", "repo names are not allowed to be '@.'");
    assertNotValid("..", "repo names are not allowed to be '@..'");
    assertNotValid("foo/bar", "repo names may contain only A-Z, a-z, 0-9, '-', '_', '.' and '~'");
    assertNotValid("foo@", "repo names may contain only A-Z, a-z, 0-9, '-', '_', '.' and '~'");
    assertNotValid("foo\0", "repo names may contain only A-Z, a-z, 0-9, '-', '_', '.' and '~'");
  }

  @Test
  public void validateUserProvidedRepoName() throws Exception {
    RepositoryName.validateUserProvidedRepoName("foo");
    RepositoryName.validateUserProvidedRepoName("foo_bar");
    RepositoryName.validateUserProvidedRepoName("foo-bar");
    RepositoryName.validateUserProvidedRepoName("foo.bar");
    RepositoryName.validateUserProvidedRepoName("foo..");
    RepositoryName.validateUserProvidedRepoName("foo.33");

    assertThrows(EvalException.class, () -> RepositoryName.validateUserProvidedRepoName(".foo"));
    assertThrows(EvalException.class, () -> RepositoryName.validateUserProvidedRepoName("_foo"));
    assertThrows(EvalException.class, () -> RepositoryName.validateUserProvidedRepoName("foo/bar"));
    assertThrows(EvalException.class, () -> RepositoryName.validateUserProvidedRepoName("@foo"));
  }

  @Test
  public void testRunfilesDir() throws Exception {
    assertThat(RepositoryName.create("foo").getRunfilesPath())
        .isEqualTo(PathFragment.create("../foo"));
    assertThat(RepositoryName.create("").getRunfilesPath()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void testGetDefaultCanonicalForm() throws Exception {
    assertThat(RepositoryName.create("").getCanonicalForm()).isEqualTo("");
    assertThat(RepositoryName.create("foo").getCanonicalForm()).isEqualTo("@foo");
  }

  @Test
  public void testGetDisplayForm() throws Exception {
    RepositoryMapping repositoryMapping =
        RepositoryMapping.create(
            ImmutableMap.of("local", RepositoryName.create("canonical")), RepositoryName.MAIN);

    assertThat(RepositoryName.create("").getDisplayForm(repositoryMapping)).isEmpty();
    assertThat(RepositoryName.create("canonical").getDisplayForm(repositoryMapping))
        .isEqualTo("@local");
    assertThat(RepositoryName.create("other").getDisplayForm(repositoryMapping))
        .isEqualTo("@@other");

    assertThat(
            RepositoryName.create("")
                .toNonVisible(RepositoryName.create("owner"))
                .getDisplayForm(repositoryMapping))
        .isEqualTo("@@[unknown repo '' requested from @owner]");
    assertThat(
            RepositoryName.create("local")
                .toNonVisible(RepositoryName.create("owner"))
                .getDisplayForm(repositoryMapping))
        .isEqualTo("@@[unknown repo 'local' requested from @owner]");
  }
}
