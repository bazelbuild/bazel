// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.skyframe.PrepareTestSuitesUnderDirectoryValue.Key;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * {@link SkyFunction} to recursively traverse a directory and ensure {@link
 * CollectTestSuitesInPackageFunction} are evaluated at each package.
 */
public class PrepareTestSuitesUnderDirectoryFunction implements SkyFunction {
  private final BlazeDirectories directories;

  PrepareTestSuitesUnderDirectoryFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    Key argument = (Key) skyKey.argument();
    ProcessPackageDirectory processPackageDirectory =
        new ProcessPackageDirectory(directories, PrepareTestSuitesUnderDirectoryValue::key);
    ProcessPackageDirectoryResult packageExistenceAndSubdirDeps =
        processPackageDirectory.getPackageExistenceAndSubdirDeps(
            argument.getRootedPath(), argument.getRepository(), env, argument.getExcludedPaths());
    if (env.valuesMissing()) {
      return null;
    }
    Iterable<SkyKey> keysToRequest = packageExistenceAndSubdirDeps.getChildDeps();
    if (packageExistenceAndSubdirDeps.packageExists()) {
      keysToRequest =
          Iterables.concat(
              ImmutableList.of(
                  CollectTestSuitesInPackageValue.key(
                      PackageIdentifier.create(
                          argument.getRepository(),
                          argument.getRootedPath().getRootRelativePath()))),
              keysToRequest);
    }
    env.getValuesOrThrow(keysToRequest, NoSuchPackageException.class);
    if (env.valuesMissing()) {
      return null;
    }
    return PrepareTestSuitesUnderDirectoryValue.INSTANCE;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
