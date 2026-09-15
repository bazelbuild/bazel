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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A source for generating unique action log paths.
 */
public final class ActionLogBufferPathGenerator {

  private final AtomicInteger actionCounter = new AtomicInteger();

  private final Path actionOutputRoot;

  public ActionLogBufferPathGenerator(Path actionOutputRoot) {
    this.actionOutputRoot = actionOutputRoot;
  }

  /**
   * Generates a unique filename for an action to store its output.
   */
  public FileOutErr generate(ArtifactPathResolver resolver) {
    int actionId = actionCounter.incrementAndGet();
    return new FileOutErr(
        resolver.convertPath(actionOutputRoot.getRelative("stdout-" + actionId)),
        resolver.convertPath(actionOutputRoot.getRelative("stderr-" + actionId)));
  }
}
