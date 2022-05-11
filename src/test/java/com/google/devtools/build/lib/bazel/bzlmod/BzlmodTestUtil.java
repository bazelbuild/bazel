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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Attribute;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map;
import java.util.Map.Entry;
import net.starlark.java.eval.Dict;
import net.starlark.java.syntax.Location;

/** Utilities for bzlmod tests. */
public final class BzlmodTestUtil {
  private BzlmodTestUtil() {}

  /** Simple wrapper around {@link ModuleKey#create} that takes a string version. */
  public static ModuleKey createModuleKey(String name, String version) {
    try {
      return ModuleKey.create(name, Version.parse(version));
    } catch (Version.ParseException e) {
      throw new IllegalArgumentException(e);
    }
  }

  /** Builder class to create a {@code Entry<ModuleKey, Module>} entry faster inside UnitTests */
  static final class ModuleBuilder {
    Module.Builder builder;
    ModuleKey key;
    ImmutableMap.Builder<String, ModuleKey> deps = new ImmutableMap.Builder<>();
    ImmutableMap.Builder<String, ModuleKey> originalDeps = new ImmutableMap.Builder<>();

    private ModuleBuilder() {}

    public static ModuleBuilder create(String name, Version version, int compatibilityLevel) {
      ModuleBuilder moduleBuilder = new ModuleBuilder();
      ModuleKey key = ModuleKey.create(name, version);
      moduleBuilder.key = key;
      moduleBuilder.builder =
          Module.builder()
              .setName(name)
              .setVersion(version)
              .setKey(key)
              .setCompatibilityLevel(compatibilityLevel);
      return moduleBuilder;
    }

    public static ModuleBuilder create(String name, String version, int compatibilityLevel)
        throws ParseException {
      return create(name, Version.parse(version), compatibilityLevel);
    }

    public static ModuleBuilder create(String name, String version) throws ParseException {
      return create(name, Version.parse(version), 0);
    }

    public static ModuleBuilder create(String name, Version version) throws ParseException {
      return create(name, version, 0);
    }

    public ModuleBuilder addDep(String depRepoName, ModuleKey key) {
      deps.put(depRepoName, key);
      return this;
    }

    public ModuleBuilder addOriginalDep(String depRepoName, ModuleKey key) {
      originalDeps.put(depRepoName, key);
      return this;
    }

    public ModuleBuilder setKey(ModuleKey value) {
      this.key = value;
      this.builder.setKey(value);
      return this;
    }

    public ModuleBuilder setRegistry(FakeRegistry value) {
      this.builder.setRegistry(value);
      return this;
    }

    public ModuleBuilder setExecutionPlatformsToRegister(ImmutableList<String> value) {
      this.builder.setExecutionPlatformsToRegister(value);
      return this;
    }

    public ModuleBuilder setToolchainsToRegister(ImmutableList<String> value) {
      this.builder.setToolchainsToRegister(value);
      return this;
    }

    public ModuleBuilder addExtensionUsage(ModuleExtensionUsage value) {
      this.builder.addExtensionUsage(value);
      return this;
    }

    public Map.Entry<ModuleKey, Module> buildEntry() {
      Module module = this.build();
      return new SimpleEntry<>(this.key, module);
    }

    public Module build() {
      ImmutableMap<String, ModuleKey> builtDeps = this.deps.buildOrThrow();

      /* Copy dep entries that have not been changed to original deps */
      ImmutableMap<String, ModuleKey> initOriginalDeps = this.originalDeps.buildOrThrow();
      for (Entry<String, ModuleKey> e : builtDeps.entrySet()) {
        if (!initOriginalDeps.containsKey(e.getKey())) {
          originalDeps.put(e);
        }
      }
      ImmutableMap<String, ModuleKey> builtOriginalDeps = this.originalDeps.buildOrThrow();

      return this.builder.setDeps(builtDeps).setOriginalDeps(builtOriginalDeps).build();
    }
  }

  public static RepositoryMapping createRepositoryMapping(ModuleKey key, String... names) {
    ImmutableMap.Builder<RepositoryName, RepositoryName> mappingBuilder = ImmutableMap.builder();
    for (int i = 0; i < names.length; i += 2) {
      mappingBuilder.put(
          RepositoryName.createUnvalidated(names[i]),
          RepositoryName.createUnvalidated(names[i + 1]));
    }
    return RepositoryMapping.create(mappingBuilder.buildOrThrow(), key.getCanonicalRepoName());
  }

  public static TagClass createTagClass(Attribute... attrs) {
    return TagClass.create(ImmutableList.copyOf(attrs), "doc", Location.BUILTIN);
  }

  /** A builder for {@link Tag} for testing purposes. */
  public static class TestTagBuilder {
    private final Dict.Builder<String, Object> attrValuesBuilder = Dict.builder();
    private final String tagName;

    private TestTagBuilder(String tagName) {
      this.tagName = tagName;
    }

    public TestTagBuilder addAttr(String attrName, Object attrValue) {
      attrValuesBuilder.put(attrName, attrValue);
      return this;
    }

    public Tag build() {
      return Tag.builder()
          .setTagName(tagName)
          .setLocation(Location.BUILTIN)
          .setAttributeValues(attrValuesBuilder.buildImmutable())
          .build();
    }
  }

  public static TestTagBuilder buildTag(String tagName) throws Exception {
    return new TestTagBuilder(tagName);
  }
}
