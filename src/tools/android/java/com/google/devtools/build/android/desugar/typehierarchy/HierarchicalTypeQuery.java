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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.util.ArrayDeque;
import java.util.LinkedHashSet;
import java.util.Queue;
import javax.annotation.Nullable;

/**
 * The entry point of a class or an interface query under the context of a type inheritance
 * hierarchy.
 */
@AutoValue
public abstract class HierarchicalTypeQuery {

  abstract HierarchicalTypeKey type();

  abstract TypeHierarchy typeHierarchy();

  public static HierarchicalTypeQuery create(
      HierarchicalTypeKey type, TypeHierarchy typeHierarchy) {
    return new AutoValue_HierarchicalTypeQuery(type, typeHierarchy);
  }

  public final boolean hasSuperClass() {
    return findDirectSuperClass() != null;
  }

  @Nullable
  public final HierarchicalTypeKey findDirectSuperClass() {
    return typeHierarchy().getDirectSuperClass(type());
  }

  public final ImmutableSet<HierarchicalTypeKey> findDirectSuperInterfaces() {
    return typeHierarchy().getDirectSuperInterfaces(type());
  }

  public final ImmutableSet<HierarchicalTypeKey> findDirectSuperTypes() {
    HierarchicalTypeKey directSuperClass = findDirectSuperClass();
    return directSuperClass == null
        ? findDirectSuperInterfaces()
        : ImmutableSet.<HierarchicalTypeKey>builder()
            .add(directSuperClass)
            .addAll(findDirectSuperInterfaces())
            .build();
  }

  public final ImmutableList<HierarchicalTypeKey> findTransitiveSuperClasses() {
    ImmutableList.Builder<HierarchicalTypeKey> superClassesBuilder = ImmutableList.builder();
    HierarchicalTypeKey workingKey = findDirectSuperClass();
    while (workingKey != null) {
      superClassesBuilder.add(workingKey);
      workingKey = workingKey.inTypeHierarchy(typeHierarchy()).findDirectSuperClass();
    }
    return superClassesBuilder.build();
  }

  public final ImmutableList<HierarchicalTypeKey> findTransitiveSuperInterfaces() {
    LinkedHashSet<HierarchicalTypeKey> transitiveInterfaces =
        new LinkedHashSet<>(findDirectSuperInterfaces());
    Queue<HierarchicalTypeKey> workingTypes = new ArrayDeque<>(findDirectSuperTypes());
    while (!workingTypes.isEmpty()) {
      HierarchicalTypeKey frontKey = workingTypes.remove();
      for (HierarchicalTypeKey superInterface :
          frontKey.inTypeHierarchy(typeHierarchy()).findDirectSuperInterfaces()) {
        if (transitiveInterfaces.add(superInterface)) {
          workingTypes.add(superInterface);
        }
      }
      HierarchicalTypeKey superClass =
          frontKey.inTypeHierarchy(typeHierarchy()).findDirectSuperClass();
      if (superClass != null) {
        workingTypes.add(superClass);
      }
    }
    return ImmutableList.copyOf(transitiveInterfaces);
  }

  public final ImmutableList<HierarchicalTypeKey> findTransitiveSuperTypes() {
    LinkedHashSet<HierarchicalTypeKey> transitiveSuperTypes =
        new LinkedHashSet<>(findDirectSuperTypes());
    Queue<HierarchicalTypeKey> workingTypes = new ArrayDeque<>(findDirectSuperTypes());
    while (!workingTypes.isEmpty()) {
      HierarchicalTypeKey frontKey = workingTypes.remove();
      for (HierarchicalTypeKey superType :
          frontKey.inTypeHierarchy(typeHierarchy()).findDirectSuperTypes()) {
        if (transitiveSuperTypes.add(superType)) {
          workingTypes.add(superType);
        }
      }
    }
    return ImmutableList.copyOf(transitiveSuperTypes);
  }

  public final ImmutableSet<HeadlessMethodKey> listMethods() {
    return typeHierarchy().getMethods(type());
  }

  public final boolean hasMethod(HeadlessMethodKey headlessMethodKey) {
    return listMethods().contains(headlessMethodKey);
  }
}
