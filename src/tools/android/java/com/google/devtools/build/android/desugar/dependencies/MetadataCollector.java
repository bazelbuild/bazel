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
package com.google.devtools.build.android.desugar.dependencies;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.android.desugar.DependencyCollector;
import com.google.devtools.build.android.desugar.proto.DesugarDeps;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.Dependency;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.DesugarDepsInfo;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.InterfaceDetails;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.InterfaceWithCompanion;
import javax.annotation.Nullable;

/** Dependency collector that emits collected metadata as a {@link DesugarDepsInfo} proto. */
public final class MetadataCollector implements DependencyCollector {

  private final boolean tolerateMissingDeps;
  private final DesugarDepsInfo.Builder info = DesugarDeps.DesugarDepsInfo.newBuilder();

  public MetadataCollector(boolean tolerateMissingDeps) {
    this.tolerateMissingDeps = tolerateMissingDeps;
  }

  @Override
  public void assumeCompanionClass(String origin, String target) {
    checkArgument(
        target.endsWith(INTERFACE_COMPANION_SUFFIX),
        "target not a companion: %s -> %s",
        origin,
        target);
    info.addAssumePresent(
        Dependency.newBuilder().setOrigin(wrapType(origin)).setTarget(wrapType(target)));
  }

  @Override
  public void missingImplementedInterface(String origin, String target) {
    checkArgument(
        !target.endsWith(INTERFACE_COMPANION_SUFFIX),
        "target seems to be a companion: %s -> %s",
        origin,
        target);
    checkState(
        tolerateMissingDeps,
        "Couldn't find interface %s on the classpath for desugaring %s",
        target,
        origin);
    info.addMissingInterface(
        Dependency.newBuilder().setOrigin(wrapType(origin)).setTarget(wrapType(target)));
  }

  @Override
  public void recordExtendedInterfaces(String origin, String... targets) {
    if (targets.length > 0) {
      InterfaceDetails.Builder details = InterfaceDetails.newBuilder().setOrigin(wrapType(origin));
      for (String target : targets) {
        details.addExtendedInterface(wrapType(target));
      }
      info.addInterfaceWithSupertypes(details);
    }
  }

  @Override
  public void recordDefaultMethods(String origin, int count) {
    checkArgument(
        !origin.endsWith(INTERFACE_COMPANION_SUFFIX), "seems to be a companion: %s", origin);
    info.addInterfaceWithCompanion(
        InterfaceWithCompanion.newBuilder()
            .setOrigin(wrapType(origin))
            .setNumDefaultMethods(count));
  }

  @Override
  @Nullable
  public byte[] toByteArray() {
    DesugarDepsInfo result = info.build();
    return DesugarDepsInfo.getDefaultInstance().equals(result) ? null : result.toByteArray();
  }

  private static DesugarDeps.Type wrapType(String internalName) {
    return DesugarDeps.Type.newBuilder().setBinaryName(internalName).build();
  }
}
