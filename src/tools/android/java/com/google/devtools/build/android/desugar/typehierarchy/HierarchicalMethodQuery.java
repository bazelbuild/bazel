/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.typehierarchy;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.android.desugar.langmodel.MethodDeclInfo;
import javax.annotation.Nullable;

/** The entry point of a method query under the context of a type inheritance hierarchy. */
@AutoValue
public abstract class HierarchicalMethodQuery {

  public abstract HierarchicalMethodKey method();

  public abstract TypeHierarchy typeHierarchy();

  static HierarchicalMethodQuery create(HierarchicalMethodKey method, TypeHierarchy typeHierarchy) {
    return new AutoValue_HierarchicalMethodQuery(method, typeHierarchy);
  }

  public final boolean isPresent() {
    return typeHierarchy().getMethods(method().owner()).contains(method().headlessMethod());
  }

  @Memoized
  @Nullable // only possible if typeHierarchy is not on type-resolution-complete mode.
  MethodDeclInfo methodMetaData() {
    return typeHierarchy().getMethodMetadata(method());
  }

  @Memoized
  @Nullable
  public HierarchicalMethodKey getFirstBaseClassMethod() {
    return Iterables.getFirst(getBaseClassMethods(), null);
  }

  @Memoized
  public ImmutableList<HierarchicalMethodKey> getBaseClassMethods() {
    HeadlessMethodKey headlessMethodKey = method().headlessMethod();
    return method().owner().inTypeHierarchy(typeHierarchy()).findTransitiveSuperClasses().stream()
        .map(type -> type.inTypeHierarchy(typeHierarchy()))
        .filter(typeAnalysis -> typeAnalysis.hasMethod(headlessMethodKey))
        .map(typeAnalysis -> HierarchicalMethodKey.create(typeAnalysis.type(), headlessMethodKey))
        .map(method -> method.inTypeHierarchy(typeHierarchy()))
        .filter(this::hasAccessTo)
        .map(HierarchicalMethodQuery::method)
        .collect(toImmutableList());
  }

  @Memoized
  public ImmutableList<HierarchicalMethodKey> getBaseInterfaceMethods() {
    HeadlessMethodKey headlessMethodKey = method().headlessMethod();
    return method()
        .owner()
        .inTypeHierarchy(typeHierarchy())
        .findTransitiveSuperInterfaces()
        .stream()
        .map(type -> type.inTypeHierarchy(typeHierarchy()))
        .filter(typeAnalysis -> typeAnalysis.hasMethod(headlessMethodKey))
        .map(typeAnalysis -> HierarchicalMethodKey.create(typeAnalysis.type(), headlessMethodKey))
        .collect(toImmutableList());
  }

  @Memoized
  public ImmutableList<HierarchicalMethodKey> getBaseMethods() {
    HeadlessMethodKey headlessMethodKey = method().headlessMethod();
    return method().owner().inTypeHierarchy(typeHierarchy()).findTransitiveSuperTypes().stream()
        .map(type -> type.inTypeHierarchy(typeHierarchy()))
        .filter(typeAnalysis -> typeAnalysis.hasMethod(headlessMethodKey))
        .map(typeAnalysis -> HierarchicalMethodKey.create(typeAnalysis.type(), headlessMethodKey))
        .map(method -> method.inTypeHierarchy(typeHierarchy()))
        .filter(this::hasAccessTo)
        .map(HierarchicalMethodQuery::method)
        .collect(toImmutableList());
  }

  /** Returns the java package of the enclosing class of the method. */
  public String getEnclosingPackage() {
    return this.method().owner().type().getPackageName();
  }

  /** True if the other method is visible from the current method. */
  private boolean hasAccessTo(HierarchicalMethodQuery other) {
    return other.methodMetaData() != null
        && (!other.methodMetaData().isPackageAccess()
            || this.getEnclosingPackage().equals(other.getEnclosingPackage()));
  }
}
