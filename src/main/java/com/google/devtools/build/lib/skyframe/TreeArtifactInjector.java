// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;

/** Supports metadata injection of action outputs into skyframe. */
public interface TreeArtifactInjector {

  /**
   * Injects the metadata of a tree artifact.
   *
   * <p>This can be used to save filesystem operations when the metadata is already known.
   *
   * @param output an output directory {@linkplain Artifact#isTreeArtifact tree artifact}
   * @param tree a {@link TreeArtifactValue} with the metadata of the files stored in the directory
   */
  void injectTree(SpecialArtifact output, TreeArtifactValue tree);
}
