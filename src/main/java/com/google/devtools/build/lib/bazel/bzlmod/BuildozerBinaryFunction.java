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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.repository.RepositoryUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * Resolves the buildozer binary path and root module file paths without evaluating module
 * extensions.
 *
 * <p>This is a lightweight alternative to {@link BazelModTidyFunction} for commands that only need
 * the buildozer binary path and module file paths.
 */
public class BuildozerBinaryFunction implements SkyFunction {

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    RootModuleFileValue rootModuleFileValue =
        (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
    if (rootModuleFileValue == null) {
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
                  RepositoryName.BAZEL_TOOLS, bazelToolsRepoMapping.repositoryMapping()));
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
    RootedPath buildozer;
    try {
      buildozer = RepositoryUtils.getRootedPathFromLabel(buildozerLabel, env);
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
    if (buildozer == null) {
      return null;
    }

    return new BuildozerBinaryValue(buildozer.asPath(), rootModuleFileValue.moduleFilePaths());
  }
}
