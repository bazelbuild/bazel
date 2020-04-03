/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.io;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.DependencyGraph;
import java.util.function.Predicate;

/**
 * Static utilities for finding referenced type symbols recursively according to the given filter.
 *
 * <p>For example, if Class A references Class B which references Class C, the the reachable types
 * from A is {A, B, C}.
 */
public final class FileBasedTypeReferenceClosure {

  private FileBasedTypeReferenceClosure() {}

  public static ImmutableSet<ClassName> findReachableReferencedTypes(
      ImmutableSet<ClassName> initialTypes, Predicate<ClassName> typeFilter) {
    return DependencyGraph.findAllReachableNodes(
            initialTypes.stream()
                .map(className -> FileBasedTypeReferenceNode.create(className, typeFilter))
                .collect(toImmutableSet()))
        .stream()
        .map(FileBasedTypeReferenceNode::className)
        .collect(toImmutableSet());
  }
}
