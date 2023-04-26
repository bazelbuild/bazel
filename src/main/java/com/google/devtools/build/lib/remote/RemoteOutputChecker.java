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
package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.RemoteArtifactChecker;
import com.google.devtools.build.lib.clock.Clock;
import java.util.regex.Pattern;

/** A {@link RemoteArtifactChecker} that checks the TTL of remote metadata. */
public class RemoteOutputChecker implements RemoteArtifactChecker {
  private final Clock clock;
  private final ImmutableList<Pattern> patternsToDownload;

  public RemoteOutputChecker(Clock clock, ImmutableList<Pattern> patternsToDownload) {
    this.clock = clock;
    this.patternsToDownload = patternsToDownload;
  }

  public boolean shouldDownloadFile(Artifact file) {
    checkArgument(!file.isTreeArtifact(), "file must not be a tree.");

    if (outputMatchesPattern(file)) {
      return true;
    }

    return false;
  }

  private boolean outputMatchesPattern(Artifact output) {
    for (var pattern : patternsToDownload) {
      if (pattern.matcher(output.getExecPathString()).matches()) {
        return true;
      }
    }
    return false;
  }

  @Override
  public boolean shouldTrustRemoteArtifact(Artifact file, RemoteFileArtifactValue metadata) {
    if (shouldDownloadFile(file)) {
      // If Bazel should download this file, but it does not exist locally, returns false to rerun
      // the generating action to trigger the download (just like in the normal build, when local
      // outputs are missing).
      return false;
    }

    return metadata.isAlive(clock.now());
  }
}
