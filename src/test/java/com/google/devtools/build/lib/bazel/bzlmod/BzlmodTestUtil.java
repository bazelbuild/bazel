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
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule.ResolutionReason;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
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

    @CanIgnoreReturnValue
    public ModuleBuilder addDep(String depRepoName, ModuleKey key) {
      deps.put(depRepoName, key);
      return this;
    }

    @CanIgnoreReturnValue
    public ModuleBuilder addOriginalDep(String depRepoName, ModuleKey key) {
      originalDeps.put(depRepoName, key);
      return this;
    }

    @CanIgnoreReturnValue
    public ModuleBuilder setKey(ModuleKey value) {
      this.key = value;
      this.builder.setKey(value);
      return this;
    }

    @CanIgnoreReturnValue
    public ModuleBuilder setRepoName(String value) {
      this.builder.setRepoName(value);
      return this;
    }

    @CanIgnoreReturnValue
    public ModuleBuilder setRegistry(FakeRegistry value) {
      this.builder.setRegistry(value);
      return this;
    }

    @CanIgnoreReturnValue
    public ModuleBuilder addExecutionPlatformsToRegister(ImmutableList<String> value) {
      this.builder.addExecutionPlatformsToRegister(value);
      return this;
    }

    @CanIgnoreReturnValue
    public ModuleBuilder addToolchainsToRegister(ImmutableList<String> value) {
      this.builder.addToolchainsToRegister(value);
      return this;
    }

    @CanIgnoreReturnValue
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

  /**
   * Builder helper for {@link
   * com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule}
   */
  public static final class AugmentedModuleBuilder {

    public static AugmentedModuleBuilder buildAugmentedModule(
        ModuleKey key, String name, Version version, boolean loaded) {
      AugmentedModuleBuilder myBuilder = new AugmentedModuleBuilder();
      myBuilder.key = key;
      myBuilder.builder =
          AugmentedModule.builder(key).setName(name).setVersion(version).setLoaded(loaded);
      return myBuilder;
    }

    public static AugmentedModuleBuilder buildAugmentedModule(
        String name, String version, boolean loaded) throws ParseException {
      ModuleKey key = createModuleKey(name, version);
      return buildAugmentedModule(key, name, Version.parse(version), loaded);
    }

    public static AugmentedModuleBuilder buildAugmentedModule(String name, String version)
        throws ParseException {
      ModuleKey key = createModuleKey(name, version);
      return buildAugmentedModule(key, name, Version.parse(version), true);
    }

    public static AugmentedModuleBuilder buildAugmentedModule(ModuleKey key, String name) {
      return buildAugmentedModule(key, name, key.getVersion(), true);
    }

    private AugmentedModule.Builder builder;
    private ModuleKey key;

    private AugmentedModuleBuilder() {}

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addChangedDep(
        String name, String version, String oldVersion, ResolutionReason reason) {
      this.builder
          .addDep(name, createModuleKey(name, version))
          .addUnusedDep(name, createModuleKey(name, oldVersion))
          .addDepReason(name, reason);
      return this;
    }

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addChangedDep(
        String repoName,
        String moduleName,
        String version,
        String oldVersion,
        ResolutionReason reason) {
      this.builder
          .addDep(repoName, createModuleKey(moduleName, version))
          .addUnusedDep(repoName, createModuleKey(moduleName, oldVersion))
          .addDepReason(repoName, reason);
      return this;
    }

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addDep(String name, String version) {
      this.builder
          .addDep(name, createModuleKey(name, version))
          .addDepReason(name, ResolutionReason.ORIGINAL);
      return this;
    }

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addDep(String repoName, String moduleName, String version) {
      this.builder
          .addDep(repoName, createModuleKey(moduleName, version))
          .addDepReason(repoName, ResolutionReason.ORIGINAL);
      return this;
    }

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addDependant(String name, String version) {
      this.builder.addDependant(createModuleKey(name, version));
      return this;
    }

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addDependant(ModuleKey key) {
      this.builder.addDependant(key);
      return this;
    }

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addOriginalDependant(String name, String version) {
      this.builder.addOriginalDependant(createModuleKey(name, version));
      return this;
    }

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addOriginalDependant(ModuleKey key) {
      this.builder.addOriginalDependant(key);
      return this;
    }

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addStillDependant(String name, String version) {
      this.builder.addOriginalDependant(createModuleKey(name, version));
      this.builder.addDependant(createModuleKey(name, version));
      return this;
    }

    @CanIgnoreReturnValue
    public AugmentedModuleBuilder addStillDependant(ModuleKey key) {
      this.builder.addOriginalDependant(key);
      this.builder.addDependant(key);
      return this;
    }

    public Entry<ModuleKey, AugmentedModule> buildEntry() {
      return new SimpleEntry<>(this.key, this.builder.build());
    }
  }

  public static RepositoryMapping createRepositoryMapping(ModuleKey key, String... names) {
    ImmutableMap.Builder<String, RepositoryName> mappingBuilder = ImmutableMap.builder();
    for (int i = 0; i < names.length; i += 2) {
      mappingBuilder.put(names[i], RepositoryName.createUnvalidated(names[i + 1]));
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
    private boolean devDependency = false;

    private TestTagBuilder(String tagName) {
      this.tagName = tagName;
    }

    @CanIgnoreReturnValue
    public TestTagBuilder addAttr(String attrName, Object attrValue) {
      attrValuesBuilder.put(attrName, attrValue);
      return this;
    }

    @CanIgnoreReturnValue
    public TestTagBuilder setDevDependency() {
      devDependency = true;
      return this;
    }

    public Tag build() {
      return Tag.builder()
          .setTagName(tagName)
          .setLocation(Location.BUILTIN)
          .setAttributeValues(AttributeValues.create(attrValuesBuilder.buildImmutable()))
          .setDevDependency(devDependency)
          .build();
    }
  }

  public static TestTagBuilder buildTag(String tagName) throws Exception {
    return new TestTagBuilder(tagName);
  }
}
