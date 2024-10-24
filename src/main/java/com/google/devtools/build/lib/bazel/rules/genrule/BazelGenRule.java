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

package com.google.devtools.build.lib.bazel.rules.genrule;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.genrule.GenRuleBase;
import java.util.List;

/** An implementation of genrule for Bazel. */
public final class BazelGenRule extends GenRuleBase {

  @Override
  protected ImmutableMap<Label, NestedSet<Artifact>> collectSources(
      List<? extends TransitiveInfoCollection> srcs) {
    ImmutableMap.Builder<Label, NestedSet<Artifact>> labelMap =
        ImmutableMap.builderWithExpectedSize(srcs.size());

    for (TransitiveInfoCollection dep : srcs) {
      NestedSet<Artifact> files = dep.getProvider(FileProvider.class).getFilesToBuild();
      labelMap.put(AliasProvider.getDependencyLabel(dep), files);
    }

    return labelMap.buildOrThrow();
  }
}
