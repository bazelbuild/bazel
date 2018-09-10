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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Exception to indicate that one {@link Action} has an output artifact whose path is a prefix of an
 * output of another action. Since the first path cannot be both a directory and a file, this would
 * lead to an error if both actions were executed in the same build.
 */
public class ArtifactPrefixConflictException extends Exception {
  public ArtifactPrefixConflictException(
      PathFragment firstPath, PathFragment secondPath, Label firstOwner, Label secondOwner) {
    super(
        String.format(
            "output path '%s' (belonging to %s) is a prefix of output path '%s' (belonging to %s). "
                + "These actions cannot be simultaneously present; please rename one of the output "
                + "files or build just one of them",
            firstPath, firstOwner, secondPath, secondOwner));
  }
}
