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

package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.platform.IncompatiblePlatformProviderApi;
import java.util.Comparator;
import javax.annotation.Nullable;

/**
 * Provider instance for the {@code target_compatible_with} attribute.
 *
 * <p>The presence of this provider is used to indicate that a target is incompatible with the
 * current platform. Any target that provides this will automatically be excluded from {@link
 * SkyframeAnalysisResult}'s list of configured targets.
 *
 * <p>This provider is able to keep track of _why_ the corresponding target is considered
 * incompatible. If the target is incompatible because the target platform didn't satisfy one of the
 * constraints in target_compatible_with, then all the relevant constraints are accessible via
 * {@code getConstraintsResponsibleForIncompatibility()}. If the target is incompatible because one
 * of the <code>config_setting</code> targets in target_compatible_with didn't match, then all the relevant
 * config_setting labels are accessible via {@code getConfigSettingsResponsibleForIncompatibility()}.
 * On the other hand, if the corresponding target is incompatible because one of its dependencies is
 * incompatible, then all the incompatible dependencies are available via {@code
 * getTargetResponsibleForIncompatibility()}.
 *
 * @param targetPlatform Returns the target platform of the target that was incompatible.
 * @param targetsResponsibleForIncompatibility Returns the incompatible dependencies that caused
 *     this provider to be present.
 *     <p>This may be null. If it is null, then at least one of {@code
 *     getConstraintsResponsibleForIncompatibility()} and {@code
 *     getConfigSettingsResponsibleForIncompatibility()} is guaranteed to be non-null. It will have
 *     at least one element in it if it is not null.
 * @param constraintsResponsibleForIncompatibility Returns the constraints that the target platform
 *     didn't satisfy.
 *     <p>This may be null. It will have at least one element in it if it is not null.
 *     <p>The list is sorted based on the stringified label of each constraint.
 * @param configSettingsResponsibleForIncompatibility Returns the config_settings that didn't match.
 *     <p>This may be null. It will have at least one element in it if it is not null.
 *     <p>The list is sorted based on the stringified label of each config_setting.
 */
@Immutable
@AutoCodec
public record IncompatiblePlatformProvider(
    @Nullable Label targetPlatform,
    @Nullable ImmutableList<ConfiguredTarget> targetsResponsibleForIncompatibility,
    @Nullable ImmutableList<ConstraintValueInfo> constraintsResponsibleForIncompatibility,
    @Nullable ImmutableList<Label> configSettingsResponsibleForIncompatibility)
    implements Info, IncompatiblePlatformProviderApi {
  /** Name used in Starlark for accessing this provider. */
  public static final String STARLARK_NAME = "IncompatiblePlatformProvider";

  /** Provider singleton constant. */
  public static final BuiltinProvider<IncompatiblePlatformProvider> PROVIDER =
      new BuiltinProvider<IncompatiblePlatformProvider>(
          STARLARK_NAME, IncompatiblePlatformProvider.class) {};

  @Override
  public BuiltinProvider<IncompatiblePlatformProvider> getProvider() {
    return PROVIDER;
  }

  public static IncompatiblePlatformProvider incompatibleDueToTargets(
      @Nullable Label targetPlatform,
      ImmutableList<ConfiguredTarget> targetsResponsibleForIncompatibility) {
    Preconditions.checkNotNull(targetsResponsibleForIncompatibility);
    Preconditions.checkArgument(!targetsResponsibleForIncompatibility.isEmpty());
    return new IncompatiblePlatformProvider(
        targetPlatform, targetsResponsibleForIncompatibility, null, null);
  }

  public static IncompatiblePlatformProvider incompatibleDueToConstraints(
      @Nullable Label targetPlatform, ImmutableList<ConstraintValueInfo> constraints) {
    return incompatibleDueToConstraintsAndConfigSettings(
        targetPlatform, constraints, ImmutableList.of());
  }

  public static IncompatiblePlatformProvider incompatibleDueToConfigSettings(
      @Nullable Label targetPlatform, ImmutableList<Label> configSettings) {
    return incompatibleDueToConstraintsAndConfigSettings(
        targetPlatform, ImmutableList.of(), configSettings);
  }

  public static IncompatiblePlatformProvider incompatibleDueToConstraintsAndConfigSettings(
      @Nullable Label targetPlatform,
      ImmutableList<ConstraintValueInfo> constraints,
      ImmutableList<Label> configSettings) {
    Preconditions.checkNotNull(constraints);
    Preconditions.checkNotNull(configSettings);
    Preconditions.checkArgument(!constraints.isEmpty() || !configSettings.isEmpty());

    // Deduplicate and sort the list of incompatible constraints. Doing it here means that everyone
    // inspecting this provider doesn't have to deal with it.
    constraints =
        constraints.stream()
            .sorted(Comparator.comparing(ConstraintValueInfo::label))
            .distinct()
            .collect(toImmutableList());

    configSettings =
        configSettings.stream()
            .sorted(Comparator.naturalOrder())
            .distinct()
            .collect(toImmutableList());

    return new IncompatiblePlatformProvider(
        targetPlatform,
        null,
        constraints.isEmpty() ? null : constraints,
        configSettings.isEmpty() ? null : configSettings);
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

}
