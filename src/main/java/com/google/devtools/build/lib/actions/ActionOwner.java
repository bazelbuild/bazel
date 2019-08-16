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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;

/**
 * Contains metadata used for reporting the progress and status of an action.
 *
 * <p>Morally an action's owner is the RuleConfiguredTarget instance responsible for creating it,
 * but to avoid storing heavyweight analysis objects in actions, and to avoid coupling between the
 * analysis and actions packages, the RuleConfiguredTarget provides an instance of this class.
 */
@AutoValue
@AutoCodec
@Immutable
public abstract class ActionOwner {
  /** An action owner for special cases. Usage is strongly discouraged. */
  public static final ActionOwner SYSTEM_ACTION_OWNER =
      ActionOwner.create(
          null,
          ImmutableList.<AspectDescriptor>of(),
          null,
          "system",
          "empty target kind",
          "system",
          null,
          null,
          null);

  @AutoCodec.Instantiator
  public static ActionOwner create(
      @Nullable Label label,
      ImmutableList<AspectDescriptor> aspectDescriptors,
      @Nullable Location location,
      @Nullable String mnemonic,
      @Nullable String targetKind,
      String configurationChecksum,
      @Nullable BuildConfigurationEvent configuration,
      @Nullable String additionalProgressInfo,
      @Nullable PlatformInfo executionPlatform) {
    return new AutoValue_ActionOwner(
        location,
        label,
        aspectDescriptors,
        mnemonic,
        Preconditions.checkNotNull(configurationChecksum),
        configuration,
        targetKind,
        additionalProgressInfo,
        executionPlatform);
  }

  /** Returns the location of this ActionOwner, if any; null otherwise. */
  @Nullable
  public abstract Location getLocation();

  /** Returns the label for this ActionOwner, if any; null otherwise. */
  @Nullable
  public abstract Label getLabel();

  public abstract ImmutableList<AspectDescriptor> getAspectDescriptors();

  /** Returns the configuration's mnemonic. */
  @Nullable
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

  /** Returns the target kind (rule class name) for this ActionOwner, if any; null otherwise. */
  @Nullable
  public abstract String getTargetKind();

  /**
   * Returns additional information that should be displayed in progress messages, or {@code null}
   * if nothing should be added.
   */
  @Nullable
  abstract String getAdditionalProgressInfo();

  /**
   * Returns the {@link PlatformInfo} platform this action should be executed on. If the execution
   * platform is {@code null}, then the host platform is assumed.
   */
  @Nullable
  abstract PlatformInfo getExecutionPlatform();

  ActionOwner() {}
}
