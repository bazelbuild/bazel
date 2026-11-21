// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpec;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/** Comparable inputs for looking up repositories independently of extension tags. */
final class RepoDefinitionInputs {
  @SerializationConstant
  static final SkyKey ROOT_KEY = () -> SkyFunctions.ROOT_REPO_DEFINITION_INPUTS;

  @SerializationConstant
  static final SkyKey RESOLVED_KEY = () -> SkyFunctions.RESOLVED_REPO_DEFINITION_INPUTS;

  record NonRegistryRepo(@Nullable String apparentName, RepoSpec spec) {}

  record Root(
      RepositoryMapping basicMainRepoMapping,
      ImmutableMap<RepositoryName, NonRegistryRepo> nonRegistryOverrides)
      implements SkyValue {}

  record Resolved(
      ImmutableMap<RepositoryName, RepoSpec> moduleSpecs,
      ImmutableMap<ModuleExtensionId, String> extensionUniqueNames)
      implements SkyValue {}

  /** Which entry in the resolved graph supplies this repository's definition. */
  sealed interface ResolvedRepo extends SkyValue {
    @AutoCodec
    record Module(RepoSpec spec) implements ResolvedRepo {}

    @AutoCodec
    record Extension(ModuleExtensionId id) implements ResolvedRepo {}

    enum Missing implements ResolvedRepo {
      INSTANCE
    }
  }

  @AutoCodec
  record ResolvedRepoKey(RepositoryName name) implements SkyKey {
    private static final SkyKeyInterner<ResolvedRepoKey> interner = SkyKey.newInterner();

    @AutoCodec.Instantiator
    static ResolvedRepoKey create(RepositoryName name) {
      return interner.intern(new ResolvedRepoKey(name));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.RESOLVED_REPO_DEFINITION;
    }

    @Override
    public SkyKeyInterner<ResolvedRepoKey> getSkyKeyInterner() {
      return interner;
    }
  }

  // Non-registry override repos can be needed during module resolution, so the root inputs must
  // be available without the resolved module graph.
  @Nullable
  static Root computeRoot(SkyKey key, SkyFunction.Environment env) throws InterruptedException {
    var root = (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
    if (root == null) {
      return null;
    }
    // String-valued label attributes in overrides (such as patches) can refer to the main repo
    // by its apparent name. Other repo spec labels are normally already resolved.
    RepositoryMapping basicMainRepoMapping =
        RepositoryMapping.create(
            ImmutableMap.<String, RepositoryName>builder()
                .put("", RepositoryName.MAIN)
                .put(root.module().getRepoName(), RepositoryName.MAIN)
                .buildKeepingLast(),
            RepositoryName.MAIN);
    var overrides = ImmutableMap.<RepositoryName, NonRegistryRepo>builder();
    root.nonRegistryOverrideCanonicalRepoToModuleName()
        .forEach(
            (canonicalName, moduleName) -> {
              if (canonicalName.equals(RepositoryName.BAZEL_TOOLS)) {
                // This repo uses Bazel's embedded tools and has a sentinel override without a spec.
                return;
              }
              var override = (NonRegistryOverride) root.overrides().get(moduleName);
              overrides.put(
                  canonicalName,
                  new NonRegistryRepo(
                      root.nonRegistryOverrideModuleToRepoName().get(moduleName),
                      override.repoSpec()));
            });
    return new Root(basicMainRepoMapping, overrides.buildOrThrow());
  }

  @Nullable
  static Resolved computeResolved(SkyKey key, SkyFunction.Environment env)
      throws InterruptedException {
    var graph = (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (graph == null) {
      return null;
    }
    var specs = ImmutableMap.<RepositoryName, RepoSpec>builder();
    graph
        .getCanonicalRepoNameLookup()
        .forEach(
            (repo, moduleKey) -> {
              RepoSpec spec = graph.getDepGraph().get(moduleKey).getRepoSpec();
              if (spec != null) {
                specs.put(repo, spec);
              }
            });
    return new Resolved(specs.buildOrThrow(), graph.getExtensionUniqueNames());
  }

  // A change to one module's spec should not recreate unrelated repository definitions.
  @Nullable
  static ResolvedRepo computeResolvedRepo(SkyKey key, SkyFunction.Environment env)
      throws InterruptedException {
    var resolved = (Resolved) env.getValue(RESOLVED_KEY);
    if (resolved == null) {
      return null;
    }
    RepositoryName name = ((ResolvedRepoKey) key).name();
    RepoSpec spec = resolved.moduleSpecs().get(name);
    if (spec != null) {
      return new ResolvedRepo.Module(spec);
    }
    for (var entry : resolved.extensionUniqueNames().entrySet()) {
      if (name.getName().startsWith(entry.getValue() + "+")) {
        return new ResolvedRepo.Extension(entry.getKey());
      }
    }
    return ResolvedRepo.Missing.INSTANCE;
  }

  private RepoDefinitionInputs() {}
}
