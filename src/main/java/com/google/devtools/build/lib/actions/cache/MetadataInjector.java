// Copyright 2019 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import java.util.Map;

/** Supports metadata injection of action outputs into skyframe. */
public interface MetadataInjector {

  /**
   * Injects the metadata of a file.
   *
   * <p>This can be used to save filesystem operations when the metadata is already known.
   *
   * <p>{@linkplain Artifact#isTreeArtifacts Tree artifacts} and their {@linkplain
   * Artifact#isChildOfDeclaredDirectory children} must not be passed here. Instead, they should be
   * passed to {@link #injectDirectory}.
   *
   * @param output a regular output file
   * @param metadata the file metadata
   */
  void injectFile(Artifact output, FileArtifactValue metadata);

  /**
   * Injects the metadata of a tree artifact.
   *
   * <p>This can be used to save filesystem operations when the metadata is already known.
   *
   * @param output an output directory {@linkplain Artifact#isTreeArtifact tree artifact}
   * @param children the metadata of the files stored in the directory
   */
  void injectDirectory(SpecialArtifact output, Map<TreeFileArtifact, FileArtifactValue> children);

  /**
   * Marks an {@link Artifact} as intentionally omitted.
   *
   * <p>This is used as an optimization to not download "orphaned artifacts" (=artifacts that no
   * action depends on) from a remote system.
   */
  void markOmitted(Artifact output);
}
