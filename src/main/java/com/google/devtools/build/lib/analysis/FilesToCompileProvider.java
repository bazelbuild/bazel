// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * A {@link TransitiveInfoProvider} that provides files to be built when the {@code --compile_only}
 * command line option is in effect. This is to avoid expensive build steps when the user only
 * wants a quick syntax check.
 */
@Immutable
public final class FilesToCompileProvider implements TransitiveInfoProvider {

  private final ImmutableList<Artifact> filesToCompile;

  public FilesToCompileProvider(ImmutableList<Artifact> filesToCompile) {
    this.filesToCompile = filesToCompile;
  }

  /**
   * Returns the list of artifacts to be built when the {@code --compile_only} command line option
   * is in effect.
   */
  public ImmutableList<Artifact> getFilesToCompile() {
    return filesToCompile;
  }
}
