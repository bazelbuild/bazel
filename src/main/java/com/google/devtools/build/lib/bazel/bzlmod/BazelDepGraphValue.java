// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;

/**
 * The result of running Bazel module resolution, containing the Bazel module dependency graph
 * post-version-resolution.
 */
@AutoValue
public abstract class BazelDepGraphValue implements SkyValue {
  @SerializationConstant public static final SkyKey KEY = () -> SkyFunctions.BAZEL_DEP_GRAPH;

  public static BazelDepGraphValue create(
      ImmutableMap<ModuleKey, Module> depGraph,
      ImmutableMap<RepositoryName, ModuleKey> canonicalRepoNameLookup,
      ImmutableList<AbridgedModule> abridgedModules,
      ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesTable,
      ImmutableMap<ModuleExtensionId, String> extensionUniqueNames,
      ImmutableTable<ModuleExtensionId, String, RepositoryName> repoOverrides) {
    return new AutoValue_BazelDepGraphValue(
        depGraph,
        ImmutableBiMap.copyOf(canonicalRepoNameLookup),
        abridgedModules,
        extensionUsagesTable,
        extensionUniqueNames,
        repoOverrides);
  }

  public static BazelDepGraphValue createEmptyDepGraph() {
    Module root =
        Module.builder()
            .setName("")
            .setVersion(Version.EMPTY)
            .setRepoName("")
            .setKey(ModuleKey.ROOT)
            .setExtensionUsages(ImmutableList.of())
            .setExecutionPlatformsToRegister(ImmutableList.of())
            .setToolchainsToRegister(ImmutableList.of())
            .build();

    ImmutableMap<ModuleKey, Module> emptyDepGraph = ImmutableMap.of(ModuleKey.ROOT, root);
    ImmutableMap<RepositoryName, ModuleKey> canonicalRepoNameLookup =
        ImmutableMap.of(RepositoryName.MAIN, ModuleKey.ROOT);

    return BazelDepGraphValue.create(
        emptyDepGraph,
        canonicalRepoNameLookup,
        ImmutableList.of(),
        ImmutableTable.of(),
        ImmutableMap.of(),
        ImmutableTable.of());
  }

  /**
   * The post-selection dep graph. Must have BFS iteration order, starting from the root module. For
   * any KEY in the returned map, it's guaranteed that {@code depGraph[KEY].getKey() == KEY}.
   */
  public abstract ImmutableMap<ModuleKey, Module> getDepGraph();

  /** A mapping from a canonical repo name to the key of the module backing it and back. */
  public abstract ImmutableBiMap<RepositoryName, ModuleKey> getCanonicalRepoNameLookup();

  /** All modules in the same order as {@link #getDepGraph}, but with limited information. */
  public abstract ImmutableList<AbridgedModule> getAbridgedModules();

  /**
   * All module extension usages grouped by the extension's ID and the key of the module where this
   * usage occurs. For each extension identifier ID, extensionUsagesTable[ID][moduleKey] is the
   * ModuleExtensionUsage of ID in the module keyed by moduleKey.
   */
  // Note: Equality of BazelDepGraphValue does not check for equality of the order of the rows of
  // this table, but it is tracked implicitly via the order of the abridged modules.
  public abstract ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage>
      getExtensionUsagesTable();

  /**
   * A mapping from the ID of a module extension to a unique string that serves as its "name". This
   * is not the same as the extension's declared name, as the declared name is only unique within
   * the .bzl file, whereas this unique name is guaranteed to be unique across the workspace.
   */
  public abstract ImmutableMap<ModuleExtensionId, String> getExtensionUniqueNames();

  public abstract ImmutableTable<ModuleExtensionId, String, RepositoryName> getRepoOverrides();

  /**
   * Returns the full {@link RepositoryMapping} for the given module, including repos from Bazel
   * module deps and module extensions.
   */
  public final RepositoryMapping getFullRepoMapping(ModuleKey key) {
    return getRepositoryMapping(
        key,
        getDepGraph(),
        getExtensionUsagesTable(),
        getExtensionUniqueNames(),
        getCanonicalRepoNameLookup(),
        getRepoOverrides());
  }

  static RepositoryMapping getRepositoryMapping(
      ModuleKey key,
      ImmutableMap<ModuleKey, Module> depGraph,
      ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesTable,
      ImmutableMap<ModuleExtensionId, String> extensionUniqueNames,
      ImmutableBiMap<RepositoryName, ModuleKey> canonicalRepoNameLookup,
      ImmutableTable<ModuleExtensionId, String, RepositoryName> repoOverrides) {
    ImmutableMap.Builder<String, RepositoryName> mapping = ImmutableMap.builder();
    for (Map.Entry<ModuleExtensionId, ModuleExtensionUsage> extIdAndUsage :
        extensionUsagesTable.column(key).entrySet()) {
      ModuleExtensionId extensionId = extIdAndUsage.getKey();
      ModuleExtensionUsage usage = extIdAndUsage.getValue();
      String repoNamePrefix = extensionUniqueNames.get(extensionId) + "+";
      for (ModuleExtensionUsage.Proxy proxy : usage.getProxies()) {
        for (Map.Entry<String, String> entry : proxy.getImports().entrySet()) {
          RepositoryName defaultCanonicalRepoName =
              RepositoryName.createUnvalidated(repoNamePrefix + entry.getValue());
          mapping.put(
              entry.getKey(),
              repoOverrides
                  .row(extensionId)
                  .getOrDefault(entry.getValue(), defaultCanonicalRepoName));
        }
      }
    }
    return depGraph
        .get(key)
        .getRepoMappingWithBazelDepsOnly(canonicalRepoNameLookup.inverse())
        .withAdditionalMappings(mapping.buildOrThrow());
  }
}
