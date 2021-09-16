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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * Computes the {@link RepositoryMapping} for use while loading a .bzl file for bzlmod (i.e. either
 * for module extension resolution or fetching a Starlark-defined repo rule).
 *
 * <p>This exists as a somewhat trivial function by itself in order to avoid having to reload all
 * such .bzl files just because some tiny irrelevant change happened in the module dependency graph.
 */
public class RepoMappingForBzlmodBzlLoadFunction implements SkyFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    String canonicalRepoName = ((RepositoryName) skyKey.argument()).strippedName();

    if (canonicalRepoName.equals("bazel_tools")) {
      // Special case: we're only here to get the @bazel_tools repo (for example, for
      // http_archive). This repo shouldn't have visibility into anything else (during repo
      // generation), so we just return an empty repo mapping.
      // TODO(wyv): disallow fallback.
      return RepoMappingForBzlmodBzlLoadValue.create(RepositoryMapping.ALWAYS_FALLBACK);
    }

    BazelModuleResolutionValue bazelModuleResolutionValue =
        (BazelModuleResolutionValue) env.getValue(BazelModuleResolutionValue.KEY);
    if (bazelModuleResolutionValue == null) {
      return null;
    }
    ModuleKey moduleKey =
        bazelModuleResolutionValue.getCanonicalRepoNameLookup().get(canonicalRepoName);
    Objects.requireNonNull(moduleKey, "Internal error: unknown repo " + canonicalRepoName);
    return RepoMappingForBzlmodBzlLoadValue.create(
        bazelModuleResolutionValue.getDepGraph().get(moduleKey).getRepoMappingWithBazelDepsOnly());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
