// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Definition of a symlink in the output tree of a Fileset rule.
 *
 * @param name location of the symlink relative to the fileset's root directory
 * @param target artifact that the symlink points to
 * @param metadata metadata of the target artifact
 */
@AutoCodec
public record FilesetOutputSymlink(PathFragment name, Artifact target, FileArtifactValue metadata) {
  public FilesetOutputSymlink {
    checkNotNull(name, "name");
    checkNotNull(target, "target");
    checkNotNull(metadata, "metadata");
    checkNotNull(metadata.getDigest(), "digest of %s", metadata);
    checkArgument(!metadata.getType().isDirectory(), "Illegal directory: %s", target);
    checkArgument(
        !target.isTreeArtifact() && !target.isFileset() && !target.isRunfilesTree(),
        "Illegal expansion artifact: %s",
        target);
  }
}
