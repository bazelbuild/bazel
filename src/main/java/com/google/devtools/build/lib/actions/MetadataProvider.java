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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.io.IOException;
import javax.annotation.Nullable;

/** Provides {@link ActionInput} metadata. */
@ThreadSafe
public interface MetadataProvider {

  /**
   * Returns a {@link FileArtifactValue} for the given {@link ActionInput}.
   *
   * <p>If the given input is an output {@link Artifact} of an action, then the returned value is
   * current as of the action's most recent execution time (which may be from a prior build, in
   * which case the value may or may not be up to date for the current build). The returned value
   * can vary across calls, for example if the action executes between calls and produces different
   * outputs than its previous execution.
   *
   * <p>The returned {@link FileArtifactValue} instance corresponds to the final target of a symlink
   * and therefore must not have a type of {@link FileStateType#SYMLINK}.
   *
   * @param input the input to retrieve the digest for
   * @return the artifact's digest or null if digest cannot be obtained (due to artifact
   *     non-existence, lookup errors, or any other reason)
   * @throws IOException if the input cannot be digested
   */
  @Nullable
  FileArtifactValue getMetadata(ActionInput input) throws IOException;

  /** Looks up an input from its exec path. */
  @Nullable
  ActionInput getInput(String execPath);
}
