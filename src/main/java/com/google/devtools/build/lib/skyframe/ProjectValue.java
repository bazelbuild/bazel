// Copyright 2024 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.Objects;
import javax.annotation.Nullable;

/** A SkyValue representing the parsed definitions from a PROJECT.scl file. */
public final class ProjectValue implements SkyValue {

  private final ImmutableSet<String> ownedCodePaths;

  private final ImmutableMap<String, Collection<String>> activeDirectories;

  private final ImmutableMap<String, Object> residualGlobals;

  public ProjectValue(
      ImmutableSet<String> ownedCodePaths,
      ImmutableMap<String, Collection<String>> activeDirectories,
      ImmutableMap<String, Object> residualGlobals) {
    this.ownedCodePaths = ownedCodePaths;
    this.residualGlobals = residualGlobals;
    this.activeDirectories = activeDirectories;
  }

  /**
   * Returns the residual global referenced by the {@code key} found in the PROJECT file.
   *
   * <p>This returns null for non-existent keys and reserved globals. Use the dedicated getters to
   * access the reserved globals. See {@code ProjectFunction.ReservedGlobals} for the list.
   */
  @Nullable
  public Object getResidualGlobal(String key) {
    return residualGlobals.get(key);
  }

  /**
   * Returns the list of code paths owned by the project. If the list is not defined in the file,
   * returns an empty set.
   */
  public ImmutableSet<String> getOwnedCodePaths() {
    return ownedCodePaths;
  }

  /**
   * Returns the map of named active directories in the project. If the map is not defined in the
   * file, returns an empty map.
   */
  public ImmutableMap<String, Collection<String>> getActiveDirectories() {
    return activeDirectories;
  }

  /** The SkyKey. Uses the label of the project file as the input. */
  public static final class Key implements SkyKey {
    private final Label projectFile;

    public Key(Label projectFile) {
      this.projectFile = Preconditions.checkNotNull(projectFile);
    }

    public Label getProjectFile() {
      return projectFile;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PROJECT;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      ProjectValue.Key key = (ProjectValue.Key) o;
      return Objects.equals(projectFile, key.projectFile);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(projectFile);
    }
  }
}
