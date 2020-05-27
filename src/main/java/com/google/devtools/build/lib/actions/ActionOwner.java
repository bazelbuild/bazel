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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.syntax.Location;
import javax.annotation.Nullable;

/**
 * Contains metadata used for reporting the progress and status of an action.
 *
 * <p>Morally an action's owner is the RuleConfiguredTarget instance responsible for creating it,
 * but to avoid storing heavyweight analysis objects in actions, and to avoid coupling between the
 * analysis and actions packages, the RuleConfiguredTarget provides an instance of this class.
 */
@AutoValue
@Immutable
public abstract class ActionOwner {
  /** An action owner for special cases. Usage is strongly discouraged. */
  @SerializationConstant
  public static final ActionOwner SYSTEM_ACTION_OWNER =
      ActionOwner.createInternal(
          null,
          ImmutableList.of(),
          Location.BUILTIN,
          "system",
          "empty target kind",
          "system",
          null,
          null,
          ImmutableMap.of(),
          null);

  public static ActionOwner create(
      Label label,
      ImmutableList<AspectDescriptor> aspectDescriptors,
      Location location,
      String mnemonic,
      String targetKind,
      String configurationChecksum,
      BuildConfigurationEvent configuration,
      @Nullable String additionalProgressInfo,
      ImmutableMap<String, String> execProperties,
      @Nullable PlatformInfo executionPlatform) {
    return createInternal(
        checkNotNull(label),
        aspectDescriptors,
        checkNotNull(location),
        checkNotNull(mnemonic),
        checkNotNull(targetKind),
        checkNotNull(configurationChecksum),
        checkNotNull(configuration),
        additionalProgressInfo,
        execProperties,
        executionPlatform);
  }

  private static ActionOwner createInternal(
      @Nullable Label label,
      ImmutableList<AspectDescriptor> aspectDescriptors,
      Location location,
      String mnemonic,
      String targetKind,
      String configurationChecksum,
      @Nullable BuildConfigurationEvent configuration,
      @Nullable String additionalProgressInfo,
      ImmutableMap<String, String> execProperties,
      @Nullable PlatformInfo executionPlatform) {
    return new AutoValue_ActionOwner(
        location,
        label,
        aspectDescriptors,
        mnemonic,
        checkNotNull(configurationChecksum),
        configuration,
        targetKind,
        additionalProgressInfo,
        execProperties,
        executionPlatform);
  }

  /** Returns the location of this ActionOwner. */
  public abstract Location getLocation();

  /** Returns the label for this ActionOwner, or null if the {@link #SYSTEM_ACTION_OWNER}. */
  @Nullable
  public abstract Label getLabel();

  public abstract ImmutableList<AspectDescriptor> getAspectDescriptors();

  /** Returns the configuration's mnemonic. */
  public abstract String getMnemonic();

  /**
   * Returns the short cache key for the configuration of the action owner.
   *
   * <p>Special action owners that are not targets can return any string here. If the underlying
   * configuration is null, this should return "null".
   */
  public abstract String getConfigurationChecksum();

  /**
   * Return the {@link BuildConfigurationEvent} associated with the action owner, if any, as it
   * should be reported in the build event protocol.
   */
  @Nullable
  public abstract BuildConfigurationEvent getConfiguration();

  /** Returns the target kind (rule class name) for this ActionOwner. */
  public abstract String getTargetKind();

  /**
   * Returns additional information that should be displayed in progress messages, or {@code null}
   * if nothing should be added.
   */
  @Nullable
  abstract String getAdditionalProgressInfo();

  /** Returns a String to String map containing the execution properties of this action. */
  @VisibleForTesting
  public abstract ImmutableMap<String, String> getExecProperties();

  /**
   * Returns the {@link PlatformInfo} platform this action should be executed on. If the execution
   * platform is {@code null}, then the host platform is assumed.
   */
  @VisibleForTesting
  @Nullable
  public abstract PlatformInfo getExecutionPlatform();
}
