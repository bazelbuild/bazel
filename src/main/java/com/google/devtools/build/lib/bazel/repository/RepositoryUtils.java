// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.skyframe.ActionEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** Utility methods related to repo fetching. */
public class RepositoryUtils {

  private RepositoryUtils() {}

  public static boolean isValidRepoRoot(Path directory) {
    // Keep in sync with //src/main/cpp/workspace_layout.h
    return directory.getRelative(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME).exists()
        || directory.getRelative(LabelConstants.WORKSPACE_FILE_NAME).exists()
        || directory.getRelative(LabelConstants.MODULE_DOT_BAZEL_FILE_NAME).exists()
        || directory.getRelative(LabelConstants.REPO_FILE_NAME).exists();
  }

  @Nullable
  public static RootedPath getRootedPathFromLabel(Label label, Environment env)
      throws InterruptedException, EvalException {
    SkyKey pkgSkyKey = PackageLookupValue.key(label.getPackageIdentifier());
    PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(pkgSkyKey);
    if (pkgLookupValue == null) {
      return null;
    }
    if (!pkgLookupValue.packageExists()) {
      String message = pkgLookupValue.getErrorMsg();
      if (pkgLookupValue == PackageLookupValue.NO_BUILD_FILE_VALUE) {
        message = PackageLookupFunction.explainNoBuildFileValue(label.getPackageIdentifier(), env);
      }
      throw Starlark.errorf("Unable to load package for %s: %s", label, message);
    }

    // And now for the file
    Root packageRoot = pkgLookupValue.getRoot();
    return RootedPath.toRootedPath(packageRoot, label.toPathFragment());
  }

  /**
   * Returns the values of the environment variables in the given set as an immutable map while
   * registering them as dependencies.
   */
  @Nullable
  public static ImmutableMap<String, Optional<String>> getEnvVarValues(
      Environment env, Set<String> keys) throws InterruptedException {
    Map<String, Optional<String>> environ = ActionEnvironmentFunction.getEnvironmentView(env, keys);
    if (environ == null) {
      return null;
    }
    Map<String, String> repoEnvOverride = PrecomputedValue.REPO_ENV.get(env);
    if (repoEnvOverride == null) {
      return null;
    }

    // Only depend on --repo_env values that are specified in keys.
    ImmutableMap.Builder<String, Optional<String>> repoEnv = ImmutableMap.builder();
    repoEnv.putAll(environ);
    for (String key : keys) {
      String value = repoEnvOverride.get(key);
      if (value != null) {
        repoEnv.put(key, Optional.of(value));
      }
    }
    return repoEnv.buildKeepingLast();
  }

  protected static Path getExternalRepositoryDirectory(BlazeDirectories directories) {
    return directories.getOutputBase().getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);
  }
}
