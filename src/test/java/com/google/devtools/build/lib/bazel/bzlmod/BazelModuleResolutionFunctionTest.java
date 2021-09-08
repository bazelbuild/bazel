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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BazelModuleResolutionFunction}. */
@RunWith(JUnit4.class)
public class BazelModuleResolutionFunctionTest {
  @Test
  public void createValue_basic() throws Exception {
    // Root depends on dep@1.0 and dep@2.0 at the same time with a multiple-version override.
    // Root also depends on rules_cc as a normal dep.
    // dep@1.0 depends on rules_java, which is overridden by a non-registry override (see below).
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleKey.ROOT,
                Module.builder()
                    .setName("my_root")
                    .setVersion(Version.parse("1.0"))
                    .setKey(ModuleKey.ROOT)
                    .addDep("my_dep_1", createModuleKey("dep", "1.0"))
                    .addDep("my_dep_2", createModuleKey("dep", "2.0"))
                    .addDep("rules_cc", createModuleKey("rules_cc", "1.0"))
                    .build())
            .put(
                createModuleKey("dep", "1.0"),
                Module.builder()
                    .setName("dep")
                    .setVersion(Version.parse("1.0"))
                    .setKey(createModuleKey("dep", "1.0"))
                    .addDep("rules_java", createModuleKey("rules_java", ""))
                    .build())
            .put(
                createModuleKey("dep", "2.0"),
                Module.builder()
                    .setName("dep")
                    .setVersion(Version.parse("2.0"))
                    .setKey(createModuleKey("dep", "2.0"))
                    .build())
            .put(
                createModuleKey("rules_cc", "1.0"),
                Module.builder()
                    .setName("rules_cc")
                    .setVersion(Version.parse("1.0"))
                    .setKey(createModuleKey("rules_cc", "1.0"))
                    .build())
            .put(
                createModuleKey("rules_java", ""),
                Module.builder()
                    .setName("rules_java")
                    .setVersion(Version.parse("1.0"))
                    .setKey(createModuleKey("rules_java", ""))
                    .build())
            .build();
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "dep",
                MultipleVersionOverride.create(
                    ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""),
            "rules_java", LocalPathOverride.create("bleh"));

    BazelModuleResolutionValue value =
        BazelModuleResolutionFunction.createValue(depGraph, overrides);
    assertThat(value.getCanonicalRepoNameLookup())
        .containsExactly(
            "", ModuleKey.ROOT,
            "dep.1.0", createModuleKey("dep", "1.0"),
            "dep.2.0", createModuleKey("dep", "2.0"),
            "rules_cc.1.0", createModuleKey("rules_cc", "1.0"),
            "rules_java.", createModuleKey("rules_java", ""));
    assertThat(value.getModuleNameLookup())
        .containsExactly(
            "rules_cc", createModuleKey("rules_cc", "1.0"),
            "rules_java", createModuleKey("rules_java", ""));
  }
}
