// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.PackageArgs;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

/** Contains information about the REPO.bazel file at the root of a repo. */
@AutoValue
public abstract class RepoFileValue implements SkyValue {
  public static final RepoFileValue EMPTY = of(PackageArgs.EMPTY);

  public abstract PackageArgs packageArgs();

  public static RepoFileValue of(PackageArgs packageArgs) {
    return new AutoValue_RepoFileValue(packageArgs);
  }

  public static Key key(RepositoryName repoName) {
    return new Key(repoName);
  }

  /** Key type for {@link RepoFileValue}. */
  public static class Key extends AbstractSkyKey<RepositoryName> {

    private Key(RepositoryName repoName) {
      super(repoName);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.REPO_FILE;
    }
  }
}
