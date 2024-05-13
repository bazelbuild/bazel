// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A value that solely represents the 'name' of a Bazel workspace, as defined in the WORKSPACE file.
 *
 * <p>This is a separate value with trivial change pruning so as to not necessitate a dependency
 * from every {@link PackageValue} to the //external {@link PackageValue}, since such a
 * hypothetical design would necessitate reloading all packages whenever there's a benign change to
 * the WORKSPACE file.
 */
public class WorkspaceNameValue implements SkyValue {
  @SerializationConstant @VisibleForSerialization
  static final SkyKey KEY = () -> SkyFunctions.WORKSPACE_NAME;

  private final String workspaceName;

  private WorkspaceNameValue(String workspaceName) {
    this.workspaceName = workspaceName;
  }

  /**
   * Returns the name of the workspace.
   */
  public String getName() {
    return workspaceName;
  }

  /** Returns the (singleton) {@link SkyKey} for {@link WorkspaceNameValue}s. */
  public static SkyKey key() {
    return KEY;
  }

  /** Returns a {@link WorkspaceNameValue} for a workspace with the given name. */
  public static WorkspaceNameValue withName(String workspaceName) {
    return new WorkspaceNameValue(Preconditions.checkNotNull(workspaceName));
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof WorkspaceNameValue other)) {
      return false;
    }
    return Objects.equals(workspaceName, other.workspaceName);
  }

  @Override
  public int hashCode() {
    return Objects.hash(workspaceName);
  }

  @Override
  public String toString() {
    return String.format("WorkspaceNameValue[name=%s]", workspaceName);
  }
}
