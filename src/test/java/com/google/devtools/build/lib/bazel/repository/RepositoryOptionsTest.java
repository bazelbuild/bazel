// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.RepositoryOverride;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.RepositoryOverrideConverter;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link RepositoryOptions}.
 */
@RunWith(JUnit4.class)
public class RepositoryOptionsTest {

  private final RepositoryOverrideConverter converter = new RepositoryOverrideConverter();

  @Rule
  public ExpectedException expectedException = ExpectedException.none();

  @Test
  public void testOverrideConverter() throws Exception {
    RepositoryOverride actual = converter.convert("foo=/bar");
    assertThat(actual.repositoryName())
        .isEqualTo(RepositoryName.createFromValidStrippedName("foo"));
    assertThat(actual.path()).isEqualTo(PathFragment.create("/bar"));
  }

  @Test
  public void testOverridePathWithEqualsSign() throws Exception {
    RepositoryOverride actual = converter.convert("foo=/bar=/baz");
    assertThat(actual.repositoryName())
        .isEqualTo(RepositoryName.createFromValidStrippedName("foo"));
    assertThat(actual.path()).isEqualTo(PathFragment.create("/bar=/baz"));
  }

  @Test
  public void testInvalidOverride() throws Exception {
    expectedException.expect(OptionsParsingException.class);
    expectedException.expectMessage(
        "Repository overrides must be of the form 'repository-name=path'");
    converter.convert("foo");
  }

  @Test
  public void testInvalidRepoOverride() throws Exception {
    expectedException.expect(OptionsParsingException.class);
    expectedException.expectMessage("Invalid repository name given to override");
    converter.convert("foo/bar=/baz");
  }

  @Test
  public void testInvalidPathOverride() throws Exception {
    expectedException.expect(OptionsParsingException.class);
    expectedException.expectMessage("Repository override directory must be an absolute path");
    converter.convert("foo=bar");
  }
}
