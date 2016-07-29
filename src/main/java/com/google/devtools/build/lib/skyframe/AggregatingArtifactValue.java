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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/** Value for aggregating artifacts, which must be expanded to a set of other artifacts. */
class AggregatingArtifactValue implements SkyValue {
  private final FileArtifactValue selfData;
  private final ImmutableList<Pair<Artifact, FileArtifactValue>> inputs;

  AggregatingArtifactValue(ImmutableList<Pair<Artifact, FileArtifactValue>> inputs,
      FileArtifactValue selfData) {
    this.inputs = inputs;
    this.selfData = selfData;
  }

  /** Returns the artifacts that this artifact expands to, together with their data. */
  Collection<Pair<Artifact, FileArtifactValue>> getInputs() {
    return inputs;
  }

  /** Returns the data of the artifact for this value, as computed by the action cache checker. */
  FileArtifactValue getSelfData() {
    return selfData;
  }
}
