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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * A target that provides .dwo files which can be combined into a .dwp packaging step. See
 * https://gcc.gnu.org/wiki/DebugFission for details.
 */
@Immutable
@AutoCodec
public final class CppDebugFileProvider implements TransitiveInfoProvider {
  private final NestedSet<Artifact> transitiveDwoFiles;
  private final NestedSet<Artifact> transitivePicDwoFiles;

  @AutoCodec.Instantiator
  public CppDebugFileProvider(
      NestedSet<Artifact> transitiveDwoFiles, NestedSet<Artifact> transitivePicDwoFiles) {
    this.transitiveDwoFiles = transitiveDwoFiles;
    this.transitivePicDwoFiles = transitivePicDwoFiles;
  }

  /**
   * Returns the .dwo files that should be included in this target's .dwp packaging (if this
   * target is linked) or passed through to a dependant's .dwp packaging (e.g. if this is a
   * cc_library depended on by a statically linked cc_binary).
   *
   * Assumes the corresponding link consumes .o files (vs. .pic.o files).
   */
  public NestedSet<Artifact> getTransitiveDwoFiles() {
    return transitiveDwoFiles;
  }

  /**
   * Same as above, but assumes the corresponding link consumes pic.o files.
   */
  public NestedSet<Artifact> getTransitivePicDwoFiles() {
    return transitivePicDwoFiles;
  }
}
