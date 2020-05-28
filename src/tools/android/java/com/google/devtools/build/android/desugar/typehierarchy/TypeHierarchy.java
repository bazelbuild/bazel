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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.android.desugar.langmodel.MethodDeclInfo;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/** An archive for class inheritance and overridable methods and memoizers of query results. */
@AutoValue
public abstract class TypeHierarchy {

  private final ConcurrentMap<HierarchicalTypeKey, HierarchicalTypeQuery> typeQueryResults =
      new ConcurrentHashMap<>();

  private final ConcurrentMap<HierarchicalMethodKey, HierarchicalMethodQuery> methodQueryResults =
      new ConcurrentHashMap<>();

  abstract ImmutableMap<HierarchicalTypeKey, HierarchicalTypeKey> directSuperClassByType();

  abstract ImmutableSetMultimap<HierarchicalTypeKey, HierarchicalTypeKey> directInterfacesByType();

  public abstract ImmutableSetMultimap<HierarchicalTypeKey, HeadlessMethodKey>
      headlessMethodKeysByType();

  public abstract ImmutableMap<HierarchicalMethodKey, MethodDeclInfo> methodMetadata();

  abstract boolean requireTypeResolutionComplete();

  public static TypeHierarchyBuilder builder() {
    return new AutoValue_TypeHierarchy.Builder();
  }

  @Nullable
  final HierarchicalTypeKey getDirectSuperClass(HierarchicalTypeKey type) {
    HierarchicalTypeKey superClass = directSuperClassByType().get(type);
    return HierarchicalTypeKey.SENTINEL.equals(superClass) ? null : superClass;
  }

  final ImmutableSet<HierarchicalTypeKey> getDirectSuperInterfaces(HierarchicalTypeKey type) {
    return directInterfacesByType().get(type);
  }

  final ImmutableSet<HeadlessMethodKey> getMethods(HierarchicalTypeKey type) {
    return headlessMethodKeysByType().get(type);
  }

  final MethodDeclInfo getMethodMetadata(HierarchicalMethodKey method) {
    MethodDeclInfo methodMetadata = methodMetadata().get(method);
    if (requireTypeResolutionComplete()) {
      checkNotNull(
          methodMetadata,
          "Expected method data present under type-resolution-complete mode for %s.");
    }
    return methodMetadata;
  }

  final HierarchicalTypeQuery query(HierarchicalTypeKey type) {
    return typeQueryResults.computeIfAbsent(type, this::createQuery);
  }

  final HierarchicalMethodQuery query(HierarchicalMethodKey method) {
    return methodQueryResults.computeIfAbsent(method, this::createQuery);
  }

  private HierarchicalTypeQuery createQuery(HierarchicalTypeKey type) {
    return HierarchicalTypeQuery.create(type, this);
  }

  private HierarchicalMethodQuery createQuery(HierarchicalMethodKey method) {
    return HierarchicalMethodQuery.create(method, this);
  }

  @AutoValue.Builder
  abstract static class TypeHierarchyBuilder {

    abstract ImmutableMap.Builder<HierarchicalTypeKey, HierarchicalTypeKey>
        directSuperClassByTypeBuilder();

    abstract ImmutableSetMultimap.Builder<HierarchicalTypeKey, HierarchicalTypeKey>
        directInterfacesByTypeBuilder();

    abstract ImmutableSetMultimap.Builder<HierarchicalTypeKey, HeadlessMethodKey>
        headlessMethodKeysByTypeBuilder();

    abstract ImmutableMap.Builder<HierarchicalMethodKey, MethodDeclInfo> methodMetadataBuilder();

    final TypeHierarchyBuilder putDirectSuperClass(
        HierarchicalTypeKey declaredType, HierarchicalTypeKey superclass) {
      directSuperClassByTypeBuilder().put(declaredType, superclass);
      return this;
    }

    final TypeHierarchyBuilder putDirectInterfaces(
        HierarchicalTypeKey declaredType, ImmutableSet<HierarchicalTypeKey> directInterfaces) {
      directInterfacesByTypeBuilder().putAll(declaredType, directInterfaces);
      return this;
    }

    final TypeHierarchyBuilder putMethod(MethodDeclInfo methodDecl) {
      checkState(!methodDecl.isPrivateAccess());
      HierarchicalTypeKey typeKey = HierarchicalTypeKey.create(methodDecl.owner());
      HierarchicalMethodKey methodKey = HierarchicalMethodKey.from(methodDecl.methodKey());
      headlessMethodKeysByTypeBuilder().put(typeKey, methodKey.headlessMethod());
      methodMetadataBuilder().put(methodKey, methodDecl);
      return this;
    }

    abstract TypeHierarchyBuilder setRequireTypeResolutionComplete(boolean value);

    abstract TypeHierarchy build();
  }
}
