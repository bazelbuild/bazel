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
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createRepositoryMapping;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.bzlmod.Module.WhichRepoMappings;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Module}. */
@RunWith(JUnit4.class)
public class ModuleTest {

  @Test
  public void canonicalizedTargetPatterns_good() throws Exception {
    ModuleKey key = createModuleKey("self", "1.0");
    Module module =
        Module.builder()
            .setExecutionPlatformsToRegister(ImmutableList.of("//:self_target"))
            .setToolchainsToRegister(ImmutableList.of("@root//:root_target", "@hi//:hi_target"))
            .addDep("hi", createModuleKey("hello", "2.0"))
            .addDep("root", ModuleKey.ROOT)
            .build();
    assertThat(module.getCanonicalizedExecutionPlatformsToRegister(key))
        .containsExactly("@self.1.0//:self_target")
        .inOrder();
    assertThat(module.getCanonicalizedToolchainsToRegister(key))
        .containsExactly("@//:root_target", "@hello.2.0//:hi_target")
        .inOrder();
  }

  @Test
  public void canonicalizedTargetPatterns_bad() throws Exception {
    ModuleKey key = createModuleKey("self", "1.0");
    Module module =
        Module.builder()
            .setExecutionPlatformsToRegister(ImmutableList.of("@what//:target"))
            .setToolchainsToRegister(ImmutableList.of("@hi:target"))
            .addDep("hi", createModuleKey("hello", "2.0"))
            .addDep("root", ModuleKey.ROOT)
            .build();
    assertThrows(
        ExternalDepsException.class,
        () -> module.getCanonicalizedExecutionPlatformsToRegister(key));
    assertThrows(
        ExternalDepsException.class, () -> module.getCanonicalizedToolchainsToRegister(key));
  }

  @Test
  public void withDepKeysTransformed() throws Exception {
    assertThat(
            Module.builder()
                .addDep("dep_foo", createModuleKey("foo", "1.0"))
                .addDep("dep_bar", createModuleKey("bar", "2.0"))
                .build()
                .withDepKeysTransformed(
                    key ->
                        createModuleKey(
                            key.getName() + "_new", key.getVersion().getOriginal() + ".1")))
        .isEqualTo(
            Module.builder()
                .addDep("dep_foo", createModuleKey("foo_new", "1.0.1"))
                .addDep("dep_bar", createModuleKey("bar_new", "2.0.1"))
                .build());
  }

  @Test
  public void getRepoMapping() throws Exception {
    ModuleKey key = createModuleKey("test_module", "1.0");
    Module module =
        Module.builder()
            .setName(key.getName())
            .setVersion(key.getVersion())
            .addDep("my_foo", createModuleKey("foo", "1.0"))
            .addDep("my_bar", createModuleKey("bar", "2.0"))
            .addDep("my_root", ModuleKey.ROOT)
            .addExtensionUsage(
                ModuleExtensionUsage.builder()
                    .setExtensionBzlFile("//:defs.bzl")
                    .setExtensionName("maven")
                    .setLocation(Location.BUILTIN)
                    .setImports(ImmutableBiMap.of("my_guava", "guava"))
                    .build())
            .build();
    assertThat(module.getRepoMapping(WhichRepoMappings.BAZEL_DEPS_ONLY, key))
        .isEqualTo(
            createRepositoryMapping(
                key,
                "test_module",
                "test_module.1.0",
                "my_foo",
                "foo.1.0",
                "my_bar",
                "bar.2.0",
                "my_root",
                ""));
    assertThat(module.getRepoMapping(WhichRepoMappings.WITH_MODULE_EXTENSIONS_TOO, key))
        .isEqualTo(
            createRepositoryMapping(
                key,
                "test_module",
                "test_module.1.0",
                "my_foo",
                "foo.1.0",
                "my_bar",
                "bar.2.0",
                "my_root",
                "",
                "my_guava",
                "maven.guava"));
  }

  @Test
  public void getRepoMapping_asMainModule() throws Exception {
    ModuleKey key = ModuleKey.ROOT;
    Module module =
        Module.builder()
            .setName("test_module")
            .setVersion(Version.parse("1.0"))
            .addDep("my_foo", createModuleKey("foo", "1.0"))
            .addDep("my_bar", createModuleKey("bar", "2.0"))
            .build();
    assertThat(module.getRepoMapping(WhichRepoMappings.BAZEL_DEPS_ONLY, key))
        .isEqualTo(
            createRepositoryMapping(
                key, "", "", "test_module", "", "my_foo", "foo.1.0", "my_bar", "bar.2.0"));
  }
}
