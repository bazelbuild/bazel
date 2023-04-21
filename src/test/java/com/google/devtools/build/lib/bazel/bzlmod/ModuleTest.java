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
// limitations under the License

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.buildModule;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createRepositoryMapping;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Module}. */
@RunWith(JUnit4.class)
public class ModuleTest {

  @Test
  public void getRepoMapping() throws Exception {
    ModuleKey key = createModuleKey("test_module", "1.0");
    Module module =
        buildModule("test_module", "1.0")
            .addDep("my_foo", createModuleKey("foo", "1.0"))
            .addDep("my_bar", createModuleKey("bar", "2.0"))
            .addDep("my_root", ModuleKey.ROOT)
            .build();
    assertThat(module.getRepoMappingWithBazelDepsOnly())
        .isEqualTo(
            createRepositoryMapping(
                key,
                "test_module",
                "test_module~1.0",
                "my_foo",
                "foo~1.0",
                "my_bar",
                "bar~2.0",
                "my_root",
                ""));
  }

  @Test
  public void getRepoMapping_asMainModule() throws Exception {
    Module module =
        buildModule("test_module", "1.0")
            .setKey(ModuleKey.ROOT)
            .addDep("my_foo", createModuleKey("foo", "1.0"))
            .addDep("my_bar", createModuleKey("bar", "2.0"))
            .build();
    assertThat(module.getRepoMappingWithBazelDepsOnly())
        .isEqualTo(
            createRepositoryMapping(
                ModuleKey.ROOT,
                "",
                "",
                "test_module",
                "",
                "my_foo",
                "foo~1.0",
                "my_bar",
                "bar~2.0"));
  }
}
