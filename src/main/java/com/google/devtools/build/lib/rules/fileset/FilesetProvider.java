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

package com.google.devtools.build.lib.rules.fileset;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FilesetTraversalParams;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Information needed by a Fileset to do the right thing when it depends on another Fileset.
 */
public interface FilesetProvider extends TransitiveInfoProvider {
  Artifact getFilesetInputManifest();
  PathFragment getFilesetLinkDir();

  /**
   * Returns a list of the traversals that went into this Fileset. Only used by Skyframe-native
   * filesets, so will be null if Skyframe-native filesets are not enabled.
   */
  @Nullable
  ImmutableList<FilesetTraversalParams> getTraversals();
}
