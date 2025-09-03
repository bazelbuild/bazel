// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RepositoryMapping}. */
@RunWith(JUnit4.class)
public final class RepositoryMappingTest {

  @Test
  public void neverFallback() throws Exception {
    RepositoryMapping mapping =
        RepositoryMapping.create(
            ImmutableMap.of("A", RepositoryName.create("com_foo_bar_a")),
            RepositoryName.create("fake_owner_repo"));
    assertThat(mapping.get("A")).isEqualTo(RepositoryName.create("com_foo_bar_a"));
    assertThat(mapping.get("B"))
        .isEqualTo(
            RepositoryName.create("B").toNonVisible(RepositoryName.create("fake_owner_repo")));
  }

  @Test
  public void additionalMappings_basic() throws Exception {
    RepositoryMapping mapping =
        RepositoryMapping.create(
                ImmutableMap.of("A", RepositoryName.create("com_foo_bar_a")),
                RepositoryName.create("fake_owner_repo"))
            .withAdditionalMappings(ImmutableMap.of("B", RepositoryName.create("com_foo_bar_b")));
    assertThat(mapping.get("A")).isEqualTo(RepositoryName.create("com_foo_bar_a"));
    assertThat(mapping.get("B")).isEqualTo(RepositoryName.create("com_foo_bar_b"));
    assertThat(mapping.get("C"))
        .isEqualTo(
            RepositoryName.create("C").toNonVisible(RepositoryName.create("fake_owner_repo")));
  }

  @Test
  public void additionalMappings_precedence() throws Exception {
    RepositoryMapping mapping =
        RepositoryMapping.create(
                ImmutableMap.of("A", RepositoryName.create("A1")), RepositoryName.MAIN)
            .withAdditionalMappings(ImmutableMap.of("A", RepositoryName.create("A2")));
    assertThat(mapping.get("A")).isEqualTo(RepositoryName.create("A1"));
  }

  @Test
  public void unknownRepoDidYouMean() throws LabelSyntaxException {
    RepositoryMapping mapping =
        RepositoryMapping.create(
            ImmutableMap.of("foo", RepositoryName.create("foo_internal")), RepositoryName.MAIN);
    assertThat(mapping.get("boo").getNameWithAt())
        .isEqualTo("@@[unknown repo 'boo' requested from @@ (did you mean 'foo'?)]");
  }
}
