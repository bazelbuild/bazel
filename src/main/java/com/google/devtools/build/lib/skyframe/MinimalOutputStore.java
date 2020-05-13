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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;

/**
 * Implementation of {@link OutputStore} that retains a minimal amount of output data necessary to
 * complete the build.
 *
 * <p>This store is intended for use with in-memory file systems, where aggressive caching would not
 * be worthwhile.
 */
final class MinimalOutputStore extends OutputStore {
  @Override
  void putArtifactData(Artifact artifact, FileArtifactValue value) {
    if (value.isMarkerValue()) {
      super.putArtifactData(artifact, value);
    }
  }
}
