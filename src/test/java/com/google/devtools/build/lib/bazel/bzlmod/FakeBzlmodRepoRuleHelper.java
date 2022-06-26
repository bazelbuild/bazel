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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.util.Optional;

/** A fake {@link BzlmodRepoRuleHelper} */
public final class FakeBzlmodRepoRuleHelper implements BzlmodRepoRuleHelper {

  private final ImmutableMap<RepositoryName, RepoSpec> repoSpecs;

  FakeBzlmodRepoRuleHelper(ImmutableMap<RepositoryName, RepoSpec> repoSpecs) {
    this.repoSpecs = repoSpecs;
  }

  @Override
  public Optional<RepoSpec> getRepoSpec(Environment env, RepositoryName repositoryName) {
    return Optional.ofNullable(repoSpecs.get(repositoryName));
  }
}
