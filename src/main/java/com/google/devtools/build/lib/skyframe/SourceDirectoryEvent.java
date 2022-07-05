// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Event issued when a source directory is encountered in {@link ArtifactFunction}. */
@AutoValue
public abstract class SourceDirectoryEvent implements Postable {
  public abstract PathFragment execPath();

  public static SourceDirectoryEvent create(PathFragment execPath) {
    return new AutoValue_SourceDirectoryEvent(execPath);
  }
}
