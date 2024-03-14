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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.bazel.bzlmod.BazelLockFileFunction.LOCKFILE_MODE;
import static com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction.IGNORE_DEV_DEPS;
import static com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction.MODULE_OVERRIDES;
import static com.google.devtools.build.lib.skyframe.PrecomputedValue.STARLARK_SEMANTICS;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.repository.NeedsSkyframeRestartException;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * Computes all information required for the {@code bazel mod tidy} command. The evaluation of all
 * module extensions used by the root module is triggered to, as a side effect, emit any {@link
 * RootModuleFileFixupEvent}s.
 */
public class BazelModTidyFunction implements SkyFunction {

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, SkyFunctionException {
    BazelDepGraphValue depGraphValue = (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (depGraphValue == null) {
      return null;
    }
    RepositoryMappingValue bazelToolsRepoMapping =
        (RepositoryMappingValue)
            env.getValue(RepositoryMappingValue.key(RepositoryName.BAZEL_TOOLS));
    if (bazelToolsRepoMapping == null) {
      return null;
    }
    Label buildozerLabel;
    try {
      buildozerLabel =
          Label.parseWithRepoContext(
              // This label always has the ".exe" extension, even on Unix, to get a single static
              // label that works on all platforms.
              "@buildozer_binary//:buildozer.exe",
              Label.RepoContext.of(
                  RepositoryName.BAZEL_TOOLS, bazelToolsRepoMapping.getRepositoryMapping()));
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
    RootedPath buildozer;
    try {
      buildozer = RepositoryFunction.getRootedPathFromLabel(buildozerLabel, env);
    } catch (NeedsSkyframeRestartException e) {
      return null;
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }

    ImmutableSet<SkyKey> extensionsUsedByRootModule =
        depGraphValue
            .getExtensionUsagesTable()
            .columnMap()
            .getOrDefault(ModuleKey.ROOT, ImmutableMap.of())
            .keySet()
            .stream()
            .map(SingleExtensionEvalValue::key)
            .collect(toImmutableSet());
    SkyframeLookupResult result = env.getValuesAndExceptions(extensionsUsedByRootModule);
    if (env.valuesMissing()) {
      return null;
    }
    for (SkyKey extension : extensionsUsedByRootModule) {
      try {
        result.getOrThrow(extension, ExternalDepsException.class);
      } catch (ExternalDepsException e) {
        if (e.getDetailedExitCode().getFailureDetail() == null
            || !e.getDetailedExitCode()
                .getFailureDetail()
                .getExternalDeps()
                .getCode()
                .equals(FailureDetails.ExternalDeps.Code.INVALID_EXTENSION_IMPORT)) {
          throw new BazelModTidyFunctionException(e, SkyFunctionException.Transience.PERSISTENT);
        }
        // This is an error bazel mod tidy can fix, so don't fail.
      }
    }

    return BazelModTidyValue.create(
        buildozer.asPath(),
        MODULE_OVERRIDES.get(env),
        IGNORE_DEV_DEPS.get(env),
        LOCKFILE_MODE.get(env),
        STARLARK_SEMANTICS.get(env));
  }

  static final class BazelModTidyFunctionException extends SkyFunctionException {

    BazelModTidyFunctionException(ExternalDepsException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
