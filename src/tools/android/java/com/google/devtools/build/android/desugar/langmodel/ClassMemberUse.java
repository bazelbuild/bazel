/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.langmodel;

import com.google.devtools.build.android.desugar.langmodel.ClassMemberUse.ClassMemberUseBuilder;

/**
 * Identifies the way a class member (field, method) is used, including method invocation and field
 * access.
 */
public abstract class ClassMemberUse<
        K extends ClassMemberKey<K>,
        B extends ClassMemberUseBuilder<K, B, R>,
        R extends ClassMemberUse<K, B, R>>
    implements TypeMappable<ClassMemberUse<K, B, R>>, Comparable<ClassMemberUse<K, B, R>> {

  /** The invocation kind of a method and get/put operations for a field. */
  public abstract MemberUseKind useKind();

  /** A field, method or constructor of a class. */
  public abstract K member();

  public abstract B toBuilder();

  @Override
  public abstract ClassMemberUse<K, B, R> acceptTypeMapper(TypeMapper typeMapper);

  @Override
  public final int compareTo(ClassMemberUse<K, B, R> other) {
    int methodKeyComparison = member().compareTo(other.member());
    if (methodKeyComparison != 0) {
      return methodKeyComparison;
    }

    return useKind().compareTo(other.useKind());
  }

  /** The base builder for {@link ClassMemberUse}. */
  abstract static class ClassMemberUseBuilder<
      K extends ClassMemberKey<K>,
      B extends ClassMemberUseBuilder<K, B, R>,
      R extends ClassMemberUse<K, B, R>> {

    abstract B setUseKind(MemberUseKind value);

    abstract B setMember(K value);

    public abstract R build();
  }
}
