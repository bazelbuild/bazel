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

package com.google.devtools.build.lib.analysis.platform;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import java.util.Map;

/**
 * A provider that supplied information about a specific language toolchain, including what platform
 * constraints are required for execution and for the target platform.
 */
@SkylarkModule(
  name = "ToolchainInfo",
  doc = "Provides access to data about a specific toolchain.",
  category = SkylarkModuleCategory.PROVIDER
)
@Immutable
public class ToolchainInfo extends SkylarkClassObject {

  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "ToolchainInfo";

  /** Skylark constructor and identifier for this provider. */
  public static final ClassObjectConstructor SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor(SKYLARK_NAME) {};

  /** Identifier used to retrieve this provider from rules which export it. */
  public static final SkylarkProviderIdentifier SKYLARK_IDENTIFIER =
      SkylarkProviderIdentifier.forKey(SKYLARK_CONSTRUCTOR.getKey());

  private final ClassObjectConstructor.Key toolchainConstructorKey;
  private final ImmutableList<ConstraintValueInfo> execConstraints;
  private final ImmutableList<ConstraintValueInfo> targetConstraints;

  public ToolchainInfo(
      ClassObjectConstructor.Key toolchainConstructorKey,
      Iterable<ConstraintValueInfo> execConstraints,
      Iterable<ConstraintValueInfo> targetConstraints,
      Map<String, Object> toolchainData,
      Location loc) {
    this(
        toolchainConstructorKey,
        ImmutableList.copyOf(execConstraints),
        ImmutableList.copyOf(targetConstraints),
        toolchainData,
        loc);
  }

  public ToolchainInfo(
      ClassObjectConstructor.Key toolchainConstructorKey,
      ImmutableList<ConstraintValueInfo> execConstraints,
      ImmutableList<ConstraintValueInfo> targetConstraints,
      Map<String, Object> toolchainData,
      Location loc) {
    super(
        SKYLARK_CONSTRUCTOR,
        ImmutableMap.<String, Object>builder()
            .put("toolchain_type", toolchainConstructorKey)
            .put("exec_compatible_with", execConstraints)
            .put("target_compatible_with", targetConstraints)
            .putAll(toolchainData)
            .build(),
        loc);

    this.toolchainConstructorKey = toolchainConstructorKey;
    this.execConstraints = execConstraints;
    this.targetConstraints = targetConstraints;
  }

  public ClassObjectConstructor.Key toolchainConstructorKey() {
    return toolchainConstructorKey;
  }

  @SkylarkCallable(
    name = "exec_compatible_with",
    doc = "The constraints on the execution platforms this toolchain supports.",
    structField = true
  )
  public ImmutableList<ConstraintValueInfo> execConstraints() {
    return execConstraints;
  }

  @SkylarkCallable(
    name = "target_compatible_with",
    doc = "The constraints on the target platforms this toolchain supports.",
    structField = true
  )
  public ImmutableList<ConstraintValueInfo> targetConstraints() {
    return targetConstraints;
  }
}
