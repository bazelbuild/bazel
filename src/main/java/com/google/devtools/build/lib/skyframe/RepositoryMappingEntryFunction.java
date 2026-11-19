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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionRepoMappingEntriesValue;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.Forward;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.ForwardKey;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.Inverse;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.InverseKey;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.ModuleInfo;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.ModuleInfoKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/** {@link SkyFunction} for {@link RepositoryMappingEntryValue}s. */
public final class RepositoryMappingEntryFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    return switch ((RepositoryMappingEntryValue.Key) skyKey) {
      case ForwardKey forwardKey -> {
        var mapping = getMapping(forwardKey.mappingSource(), env);
        if (env.valuesMissing()) {
          yield null;
        }
        yield mapping == null
            ? Forward.REPO_NOT_FOUND
            : Forward.of(Forward.payloadFor(mapping.entries(), forwardKey.apparentName()));
      }
      case InverseKey inverseKey -> {
        var mapping = getMapping(inverseKey.mappingSource(), env);
        if (env.valuesMissing()) {
          yield null;
        }
        yield mapping == null
            ? Inverse.NONE
            : Inverse.of(mapping.getInverse(inverseKey.canonicalName()));
      }
      case ModuleInfoKey moduleInfoKey -> computeModuleInfo(moduleInfoKey.repo(), env);
    };
  }

  /**
   * Returns a mapping with the entries of the given source, or null if the source doesn't exist or
   * values are missing.
   */
  @Nullable
  private static RepositoryMapping getMapping(Object mappingSource, Environment env)
      throws InterruptedException {
    if (mappingSource instanceof RepositoryName repo) {
      var mappingValue = (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(repo));
      return mappingValue == null ? null : mappingValue.repositoryMapping();
    }
    var entriesValue =
        (ModuleExtensionRepoMappingEntriesValue) env.getValue((SkyKey) mappingSource);
    if (entriesValue == null) {
      return null;
    }
    // Lookups don't depend on the context repo.
    return RepositoryMapping.createWithKnownEntriesHashCode(
        entriesValue.entries(), entriesValue.entriesHashCode(), RepositoryName.MAIN);
  }

  @Nullable
  private static ModuleInfo computeModuleInfo(RepositoryName repo, Environment env)
      throws InterruptedException {
    var mappingValue = (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(repo));
    if (mappingValue == null) {
      return null;
    }
    if (mappingValue.repositoryMapping() == null) {
      return ModuleInfo.REPO_NOT_FOUND;
    }
    Object mappingSource = repo;
    if (!repo.isMain() && !StarlarkBuiltinsValue.isBuiltinsRepo(repo)) {
      var depGraph = (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
      if (depGraph == null) {
        return null;
      }
      if (!depGraph.getCanonicalRepoNameLookup().containsKey(repo)) {
        var extensionId = RepositoryMappingFunction.maybeGetModuleExtensionForRepo(repo, depGraph);
        if (extensionId.isPresent()) {
          mappingSource = ModuleExtensionRepoMappingEntriesValue.key(extensionId.get());
        }
      }
    }
    return new ModuleInfo(
        true,
        mappingValue.associatedModuleName(),
        mappingValue.associatedModuleVersion(),
        mappingValue.moduleRepoName(),
        mappingSource);
  }
}
