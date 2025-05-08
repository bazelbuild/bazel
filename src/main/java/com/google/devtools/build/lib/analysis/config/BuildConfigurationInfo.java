// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.BuildConfigurationEvent;
import com.google.devtools.build.lib.actions.CommandLineLimits;
import javax.annotation.Nullable;

/**
 * Provides build configuration dependent information.
 *
 * <p>By having this interface, we don't need to create a full {@link BuildConfigurationValue}
 * instance when only the four fields defined in this interface is provided.
 *
 * <p>This provides some convenience to construct {@link
 * com.google.devtools.build.lib.actions.ActionOwner#SYSTEM_ACTION_OWNER} and other {@link
 * com.google.devtools.build.lib.actions.ActionOwner} instances in tests.
 */
public interface BuildConfigurationInfo {
  /**
   * Returns the configuration-dependent string for this configuration.
   *
   * <p>This is also the name of the configuration's base output directory. See also {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfigurationValue#getOutputDirectoryName}.
   */
  String getMnemonic();

  /** Returns the cache key of the build options used to create this configuration. */
  String checksum();

  /**
   * Returns the {@code BuildEvent} associated with {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfigurationValue}.
   */
  @Nullable
  BuildConfigurationEvent toBuildEvent();

  /** Returns true if this is a tool-related configuration. */
  boolean isToolConfiguration();

  CommandLineLimits getCommandLineLimits();

  /**
   * An auto value class of {@link BuildConfigurationInfo}. This provides a convenient way for
   * creating {@link BuildConfigurationInfo} with only the four fields provided.
   */
  @AutoValue
  abstract class AutoBuildConfigurationInfo implements BuildConfigurationInfo {
    public static AutoBuildConfigurationInfo create(
        String mnemonic,
        String checksum,
        @Nullable BuildConfigurationEvent buildConfigurationEvent,
        boolean isToolConfiguration) {
      return new AutoValue_BuildConfigurationInfo_AutoBuildConfigurationInfo(
          mnemonic, checksum, buildConfigurationEvent, isToolConfiguration);
    }

    @Override
    public final CommandLineLimits getCommandLineLimits() {
      return CommandLineLimits.UNLIMITED;
    }
  }
}
