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

import com.google.devtools.build.android.desugar.proto.DesugarDeps;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.Dependency;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.DesugarDepsInfo;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.InterfaceDetails;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.InterfaceWithCompanion;
import com.google.devtools.build.android.r8.DependencyCollector;
import java.util.ArrayList;
import java.util.Comparator;
import javax.annotation.Nullable;

/** Dependency collector that emits collected metadata as a {@link DesugarDepsInfo} proto. */
public final class MetadataCollector implements DependencyCollector {

  private final boolean tolerateMissingDeps;

  private final ArrayList<DesugarDeps.Dependency> assumePresents = new ArrayList<>();
  private final ArrayList<DesugarDeps.Dependency> missingInterfaces = new ArrayList<>();
  private final ArrayList<DesugarDeps.InterfaceDetails> interfacesWithSupertypes =
      new ArrayList<>();
  private final ArrayList<DesugarDeps.InterfaceWithCompanion> interfacesWithCompanion =
      new ArrayList<>();

  public MetadataCollector(boolean tolerateMissingDeps) {
    this.tolerateMissingDeps = tolerateMissingDeps;
  }

  private static boolean isInterfaceCompanionClass(String name) {
    return name.endsWith(INTERFACE_COMPANION_SUFFIX)
        || name.endsWith(D8_INTERFACE_COMPANION_SUFFIX);
  }

  @Override
  public void assumeCompanionClass(String origin, String target) {
    checkArgument(
        isInterfaceCompanionClass(target), "target not a companion: %s -> %s", origin, target);
    assumePresents.add(
        Dependency.newBuilder().setOrigin(wrapType(origin)).setTarget(wrapType(target)).build());
  }

  @Override
  public void missingImplementedInterface(String origin, String target) {
    checkArgument(
        !isInterfaceCompanionClass(target),
        "target seems to be a companion: %s -> %s",
        origin,
        target);
    checkState(
        tolerateMissingDeps,
        "Couldn't find interface %s on the classpath for desugaring %s",
        target,
        origin);
    missingInterfaces.add(
        Dependency.newBuilder().setOrigin(wrapType(origin)).setTarget(wrapType(target)).build());
  }

  @Override
  public void recordExtendedInterfaces(String origin, String... targets) {
    if (targets.length > 0) {
      InterfaceDetails.Builder details = InterfaceDetails.newBuilder().setOrigin(wrapType(origin));
      ArrayList<DesugarDeps.Type> types = new ArrayList<>();
      for (String target : targets) {
        types.add(wrapType(target));
      }
      types.sort(Comparator.comparing(DesugarDeps.Type::getBinaryName));
      details.addAllExtendedInterface(types);
      interfacesWithSupertypes.add(details.build());
    }
  }

  @Override
  public void recordDefaultMethods(String origin, int count) {
    checkArgument(!isInterfaceCompanionClass(origin), "seems to be a companion: %s", origin);
    interfacesWithCompanion.add(
        InterfaceWithCompanion.newBuilder()
            .setOrigin(wrapType(origin))
            .setNumDefaultMethods(count)
            .build());
  }

  @Override
  @Nullable
  public byte[] toByteArray() {
    DesugarDeps.DesugarDepsInfo result = buildInfo();
    return DesugarDeps.DesugarDepsInfo.getDefaultInstance().equals(result)
        ? null
        : result.toByteArray();
  }

  private DesugarDeps.DesugarDepsInfo buildInfo() {

    // Sort these for determinism.
    assumePresents.sort(dependencyComparator);
    missingInterfaces.sort(dependencyComparator);
    interfacesWithSupertypes.sort(interfaceDetailComparator);
    interfacesWithCompanion.sort(interFaceWithCompanionComparator);

    DesugarDeps.DesugarDepsInfo.Builder info = DesugarDeps.DesugarDepsInfo.newBuilder();
    info.addAllAssumePresent(assumePresents);
    info.addAllMissingInterface(missingInterfaces);
    info.addAllInterfaceWithSupertypes(interfacesWithSupertypes);
    info.addAllInterfaceWithCompanion(interfacesWithCompanion);

    return info.build();
  }

  private static final Comparator<? super DesugarDeps.Dependency> dependencyComparator =
      Comparator.comparing((DesugarDeps.Dependency o) -> o.getOrigin().getBinaryName())
          .thenComparing((DesugarDeps.Dependency o) -> o.getTarget().getBinaryName());

  private static final Comparator<? super DesugarDeps.InterfaceDetails> interfaceDetailComparator =
      Comparator.comparing((DesugarDeps.InterfaceDetails o) -> o.getOrigin().getBinaryName());

  private static final Comparator<? super DesugarDeps.InterfaceWithCompanion>
      interFaceWithCompanionComparator = Comparator.comparing(o -> o.getOrigin().getBinaryName());

  private static DesugarDeps.Type wrapType(String internalName) {
    return DesugarDeps.Type.newBuilder().setBinaryName(internalName).build();
  }
}
