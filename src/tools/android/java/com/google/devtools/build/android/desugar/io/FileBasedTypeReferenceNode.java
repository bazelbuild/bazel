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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.DependencyGraph.Node;
import java.util.function.Predicate;

/**
 * A graph {@link Node} that represents a Java class in complied binary.
 *
 * <p>Subject to a given type filter, the direct children represent the types with a reference from
 * the class of this parent node. This implementation relies on reading compiled class files.
 */
@AutoValue
abstract class FileBasedTypeReferenceNode implements Node<FileBasedTypeReferenceNode> {

  public static FileBasedTypeReferenceNode create(
      ClassName className, Predicate<ClassName> typeFilter) {
    return new AutoValue_FileBasedTypeReferenceNode(className, typeFilter);
  }

  abstract ClassName className();

  abstract Predicate<ClassName> typeFilter();

  @Override
  public final ImmutableSet<FileBasedTypeReferenceNode> getAllChildren() {
    return FileContentProvider.fromResources(className()).findReferencedTypes(typeFilter()).stream()
        .map(childClassName -> create(childClassName, typeFilter()))
        .collect(toImmutableSet());
  }
}
