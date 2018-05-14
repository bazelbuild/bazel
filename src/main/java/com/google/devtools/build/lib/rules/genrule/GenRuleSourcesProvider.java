// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.genrule;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A transitive info provider that will give source files to genrule. */
@Immutable
@AutoCodec
public final class GenRuleSourcesProvider implements TransitiveInfoProvider {

  private final NestedSet<Artifact> genruleFiles;

  public GenRuleSourcesProvider(NestedSet<Artifact> genruleFiles) {
    this.genruleFiles = genruleFiles;
  }

  public NestedSet<Artifact> getGenruleFiles() {
    return genruleFiles;
  }
}
