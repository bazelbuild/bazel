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

/** Tests for @{link RepositoryMapping}. */
@RunWith(JUnit4.class)
public final class RepositoryMappingTest {

  @Test
  public void maybeFallback() throws Exception {
    RepositoryMapping mapping =
        RepositoryMapping.createAllowingFallback(
            ImmutableMap.of(RepositoryName.create("A"), RepositoryName.create("com_foo_bar_a")));
    assertThat(mapping.get(RepositoryName.create("A")))
        .isEqualTo(RepositoryName.create("com_foo_bar_a"));
    assertThat(mapping.get(RepositoryName.create("B"))).isEqualTo(RepositoryName.create("B"));
  }

  @Test
  public void neverFallback() throws Exception {
    RepositoryMapping mapping =
        RepositoryMapping.create(
            ImmutableMap.of(RepositoryName.create("A"), RepositoryName.create("com_foo_bar_a")),
            "fake_owner_repo");
    assertThat(mapping.get(RepositoryName.create("A")))
        .isEqualTo(RepositoryName.create("com_foo_bar_a"));
    assertThat(mapping.get(RepositoryName.create("B")))
        .isEqualTo(RepositoryName.create("B").toNonVisible("fake_owner_repo"));
  }

  @Test
  public void additionalMappings() throws Exception {
    RepositoryMapping mapping =
        RepositoryMapping.create(
                ImmutableMap.of(RepositoryName.create("A"), RepositoryName.create("com_foo_bar_a")),
                "fake_owner_repo")
            .withAdditionalMappings(
                ImmutableMap.of(
                    RepositoryName.create("B"), RepositoryName.create("com_foo_bar_b")));
    assertThat(mapping.get(RepositoryName.create("A")))
        .isEqualTo(RepositoryName.create("com_foo_bar_a"));
    assertThat(mapping.get(RepositoryName.create("B")))
        .isEqualTo(RepositoryName.create("com_foo_bar_b"));
    assertThat(mapping.get(RepositoryName.create("C")))
        .isEqualTo(RepositoryName.create("C").toNonVisible("fake_owner_repo"));
  }
}
