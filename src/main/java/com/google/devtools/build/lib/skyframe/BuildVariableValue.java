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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.view.TopLevelArtifactContext;
import com.google.devtools.build.lib.view.WorkspaceStatusAction;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Map;
import java.util.UUID;

import javax.annotation.Nullable;

/**
 * A value that represents a global build variable.
 *
 * <p>This is basically a box for "miscellaneous" auxiliary build-global values that do not merit
 * their own value builder.
 */
public class BuildVariableValue implements SkyValue {

  static final BuildVariable<String> DEFAULTS_PACKAGE_CONTENTS =
      new BuildVariable<>(new SkyKey(SkyFunctions.BUILD_VARIABLE, "default_pkg"));

  static final BuildVariable<RuleVisibility> DEFAULT_VISIBILITY =
      new BuildVariable<>(new SkyKey(SkyFunctions.BUILD_VARIABLE, "default_visibility"));

  static final BuildVariable<UUID> BUILD_ID =
      new BuildVariable<>(new SkyKey(SkyFunctions.BUILD_VARIABLE, "build_id"));

  static final BuildVariable<WorkspaceStatusAction> WORKSPACE_STATUS_KEY =
      new BuildVariable<>(new SkyKey(SkyFunctions.BUILD_VARIABLE, "workspace_status_action"));

  static final BuildVariable<TopLevelArtifactContext> TOP_LEVEL_CONTEXT =
      new BuildVariable<>(new SkyKey(SkyFunctions.BUILD_VARIABLE, "top_level_context"));

  static final BuildVariable<Map<BuildInfoKey, BuildInfoFactory>> BUILD_INFO_FACTORIES =
      new BuildVariable<>(new SkyKey(SkyFunctions.BUILD_VARIABLE, "build_info_factories"));

  static final BuildVariable<Map<String, String>> TEST_ENVIRONMENT_VARIABLES =
      new BuildVariable<>(new SkyKey(SkyFunctions.BUILD_VARIABLE, "test_environment"));

  static final BuildVariable<BlazeDirectories> BLAZE_DIRECTORIES =
      new BuildVariable<>(new SkyKey(SkyFunctions.BUILD_VARIABLE, "blaze_directories"));

  static final BuildVariable<ImmutableMap<Action, Exception>> BAD_ACTIONS =
      new BuildVariable<>(new SkyKey(SkyFunctions.BUILD_VARIABLE, "bad_actions"));

  private final Object value;

  public BuildVariableValue(Object value) {
    this.value = Preconditions.checkNotNull(value);
  }

  /**
   * Returns the value of the variable.
   */
  public Object get() {
    return value;
  }

  @Override
  public int hashCode() {
    return value.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof BuildVariableValue)) {
      return false;
    }
    BuildVariableValue other = (BuildVariableValue) obj;
    return value.equals(other.value);
  }

  @Override
  public String toString() {
    return "<BuildVariable " + value + ">";
  }

  /**
   * A helper object corresponding to a variable in Skyframe.
   *
   * <p>Instances do not have internal state.
   */
  static final class BuildVariable<T> {
    private final SkyKey key;

    private BuildVariable(SkyKey key) {
      this.key = key;
    }

    @VisibleForTesting
    SkyKey getKeyForTesting() {
      return key;
    }

    /**
     * Retrieves the value of this variable from Skyframe.
     *
     * <p>If the value was not set, an exception will be raised.
     */
    @Nullable
    @SuppressWarnings("unchecked")
    T get(SkyFunction.Environment env) {
      BuildVariableValue value = (BuildVariableValue) env.getValue(key);
      if (value == null) {
        return null;
      }
      return (T) value.get();
    }

    /**
     * Injects a new variable value.
     */
    void set(RecordingDifferencer differencer, T value) {
      differencer.inject(ImmutableMap.of(key, (SkyValue) new BuildVariableValue(value)));
    }
  }
}
