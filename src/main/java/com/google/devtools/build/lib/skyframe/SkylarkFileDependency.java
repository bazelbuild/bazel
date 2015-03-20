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

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.syntax.Label;

import java.io.Serializable;

/**
 * A simple value class to store the direct Skylark file dependencies of a Skylark
 * extension file. It also contains a Label identifying the extension file.
 */
class SkylarkFileDependency implements Serializable {

  private final Label label;
  private final ImmutableList<SkylarkFileDependency> dependencies;

  SkylarkFileDependency(Label label, ImmutableList<SkylarkFileDependency> dependencies) {
    this.label = label;
    this.dependencies = dependencies;
  }

  /**
   * Returns the list of direct Skylark file dependencies of the Skylark extension file
   * corresponding to this object.
   */
  ImmutableList<SkylarkFileDependency> getDependencies() {
    return dependencies;
  }

  /**
   * Returns the Label of the Skylark extension file corresponding to this object.
   */
  Label getLabel() {
    return label;
  }
}
