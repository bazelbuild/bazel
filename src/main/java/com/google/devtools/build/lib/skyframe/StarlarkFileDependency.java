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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Objects;

/** A value class representing a node in a DAG that mirrors the load graph of a Starlark file. */
// TODO(adonovan): opt: eliminate this class; it is redundant w.r.t. the Module DAG.
// See comment at setStarlarkFileDependencies call in PackageFactory.
@AutoCodec
public class StarlarkFileDependency {

  private final Label label;
  private final ImmutableList<StarlarkFileDependency> dependencies;

  public StarlarkFileDependency(Label label, ImmutableList<StarlarkFileDependency> dependencies) {
    this.label = label;
    this.dependencies = dependencies;
  }

  /**
   * Returns the list of direct Starlark file dependencies of the Starlark extension file
   * corresponding to this object.
   */
  public ImmutableList<StarlarkFileDependency> getDependencies() {
    return dependencies;
  }

  /** Returns the Label of the Starlark extension file corresponding to this object. */
  public Label getLabel() {
    return label;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof StarlarkFileDependency)) {
      return false;
    }
    StarlarkFileDependency other = (StarlarkFileDependency) obj;
    if (!label.equals(other.getLabel())) {
      return false;
    }
    return dependencies.equals(other.getDependencies());
  }

  @Override
  public int hashCode() {
    return Objects.hash(label, dependencies);
  }
}
