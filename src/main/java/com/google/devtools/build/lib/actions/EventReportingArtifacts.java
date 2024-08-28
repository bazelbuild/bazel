// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions.OutputGroupFileModes;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import java.util.Collection;

/** Interface for {@link BuildEvent}s reporting artifacts as named sets */
public interface EventReportingArtifacts extends BuildEvent {

  /** Pair of artifacts and a {@link CompletionContext}. */
  class ReportedArtifacts {
    public final Collection<NestedSet<Artifact>> artifacts;
    public final CompletionContext completionContext;

    public ReportedArtifacts(
        Collection<NestedSet<Artifact>> artifacts, CompletionContext completionContext) {
      this.artifacts = artifacts;
      this.completionContext = completionContext;
    }
  }

  /** The sets of artifacts this build event assumes already known in the build event stream. */
  ReportedArtifacts reportedArtifacts(OutputGroupFileModes outputGroupFileModes);
}
