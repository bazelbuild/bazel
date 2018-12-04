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
package com.google.devtools.build.lib.actions.cache;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;

/**
 * Allows to inject metadata of action outputs into skyframe.
 */
public interface MetadataInjector {

  /**
   * Injects metadata of a file that is stored remotely.
   *
   * @param file  a regular output file.
   * @param digest  the digest of the file.
   * @param sizeBytes the size of the file in bytes.
   * @param locationIndex is only used in Blaze.
   */
  default void injectRemoteFile(Artifact file, byte[] digest, long sizeBytes, int locationIndex) {
    throw new UnsupportedOperationException();
  }

  /**
   * Inject the metadata of a tree artifact whose contents are stored remotely.
   *
   * @param treeArtifact  an output directory.
   * @param children  the metadata of the files stored in the directory.
   */
  default void injectRemoteDirectory(SpecialArtifact treeArtifact,
      Map<PathFragment, RemoteFileArtifactValue> children) {
    throw new UnsupportedOperationException();
  }
}
