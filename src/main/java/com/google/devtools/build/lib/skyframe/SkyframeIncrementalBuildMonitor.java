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

import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ChangedFilesMessage;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;

import java.util.HashSet;
import java.util.Set;

/**
 * A package-private class intended to track a small number of modified files during the build. This
 * class should stop recording changed files if there are too many of them, instead of holding onto
 * a large collection of files.
 */
@ThreadSafety.ThreadCompatible
class SkyframeIncrementalBuildMonitor {
  private static final int MAX_FILES = 100;

  private Set<PathFragment> files = new HashSet<>();
  private int fileCount;

  public void accrue(Iterable<SkyKey> invalidatedValues) {
    for (SkyKey skyKey : invalidatedValues) {
      if (skyKey.functionName().equals(SkyFunctions.FILE_STATE)) {
        RootedPath file = (RootedPath) skyKey.argument();
        maybeAddFile(file.getRelativePath());
      }
    }
  }

  private void maybeAddFile(PathFragment path) {
    if (files != null) {
      files.add(path);
      if (files.size() >= MAX_FILES) {
        files = null;
      }
    }

    fileCount++;
  }

  public void alertListeners(EventBus eventBus) {
    Set<PathFragment> changedFiles = (files != null) ? files : ImmutableSet.<PathFragment>of();
    eventBus.post(new ChangedFilesMessage(changedFiles, fileCount));
  }
}
