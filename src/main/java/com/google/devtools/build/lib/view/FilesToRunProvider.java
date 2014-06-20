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

package com.google.devtools.build.lib.view;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.syntax.Label;

import javax.annotation.Nullable;

/**
 * Returns information about executables produced by a target and the files needed to run it.
 */
public interface FilesToRunProvider extends TransitiveInfoProvider {
  /**
   * Returns the label that is associated with this piece of information.
   *
   * <p>This is usually the label of the target that provides the information.
   */
  Label getLabel();

  /**
   * Returns the executable artifact for a target.  May return null.
   */
  // TODO(bazel-team): Remove this because these are also present in getRunfilesSupport()
  @Nullable Artifact getExecutable();

  /**
   * Returns artifacts needed to run the executable for this target.
   */
  ImmutableList<Artifact> getFilesToRun();

  /**
   * Returns the runfiles manifest for the corresponding target.
   *
   * <p>Returns null if the target does not have {@link RunfilesSupport}.
   */
  // TODO(bazel-team): Remove this because these are also present in getRunfilesSupport()
  @Nullable Artifact getRunfilesManifest();

  /**
   * Returns the {@RunfilesSupport} object associated with the target or null if it does not exist.
   */
  @Nullable RunfilesSupport getRunfilesSupport();
}
