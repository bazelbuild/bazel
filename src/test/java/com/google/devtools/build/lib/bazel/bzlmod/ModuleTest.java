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

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.buildModule;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createRepositoryMapping;

import com.google.devtools.build.lib.windows.WindowsShortPath;
import java.util.stream.Stream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Module}. */
@RunWith(JUnit4.class)
public class ModuleTest {

  @Test
  public void getRepoMapping() throws Exception {
    ModuleKey key = createModuleKey("test_module", "1.0");
    ModuleKey fooKey = createModuleKey("foo", "1.0");
    ModuleKey barKey = createModuleKey("bar", "2.0");
    Module module =
        buildModule("test_module", "1.0")
            .addDep("my_foo", fooKey)
            .addDep("my_bar", barKey)
            .addDep("my_root", ModuleKey.ROOT)
            .build();
    assertThat(
            module.getRepoMappingWithBazelDepsOnly(
                Stream.of(key, fooKey, barKey, ModuleKey.ROOT)
                    .collect(
                        toImmutableMap(k -> k, ModuleKey::getCanonicalRepoNameWithoutVersionForTesting))))
        .isEqualTo(
            createRepositoryMapping(
                key,
                "test_module",
                "test_module~",
                "my_foo",
                "foo~",
                "my_bar",
                "bar~",
                "my_root",
                ""));
  }

  @Test
  public void getRepoMapping_asMainModule() throws Exception {
    ModuleKey fooKey = createModuleKey("foo", "1.0");
    ModuleKey barKey = createModuleKey("bar", "2.0");
    Module module =
        buildModule("test_module", "1.0")
            .setKey(ModuleKey.ROOT)
            .addDep("my_foo", createModuleKey("foo", "1.0"))
            .addDep("my_bar", createModuleKey("bar", "2.0"))
            .build();
    assertThat(
            module.getRepoMappingWithBazelDepsOnly(
                Stream.of(ModuleKey.ROOT, fooKey, barKey)
                    .collect(toImmutableMap(k -> k, ModuleKey::getCanonicalRepoNameWithVersionForTesting))))
        .isEqualTo(
            createRepositoryMapping(
                ModuleKey.ROOT,
                "",
                "",
                "test_module",
                "",
                "my_foo",
                "foo~v1.0",
                "my_bar",
                "bar~v2.0"));
  }

  @Test
  public void getCanonicalRepoName_isNotAWindowsShortPath() {
    assertNotAShortPath(createModuleKey("foo", "").getCanonicalRepoNameWithoutVersionForTesting().getName());
    assertNotAShortPath(createModuleKey("foo", "1").getCanonicalRepoNameWithVersionForTesting().getName());
    assertNotAShortPath(createModuleKey("foo", "1.2").getCanonicalRepoNameWithVersionForTesting().getName());
    assertNotAShortPath(
        createModuleKey("foo", "1.2.3").getCanonicalRepoNameWithVersionForTesting().getName());
  }

  private static void assertNotAShortPath(String name) {
    assertWithMessage("For %s", name).that(WindowsShortPath.isShortPath(name)).isFalse();
  }
}
