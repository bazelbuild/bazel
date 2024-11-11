// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.base.Preconditions.checkArgument;

import java.util.Arrays;

/**
 * A representation of a recursively composable set of {@link FileSystemDependencies}.
 *
 * <p>This corresponds to a previously serialized {@link
 * com.google.devtools.build.lib.skyframe.NestedFileSystemOperationNodes} instance, but this
 * implementation is mostly decoupled from Bazel code.
 */
final class NestedDependencies
    implements FileSystemDependencies, FileDependencyDeserializer.GetNestedDependenciesResult {
  private final FileSystemDependencies[] elements;

  NestedDependencies(FileSystemDependencies[] elements) {
    checkArgument(elements.length > 1, "expected at least length 2, was %s", elements.length);
    this.elements = elements;
  }

  int count() {
    return elements.length;
  }

  FileSystemDependencies getElement(int index) {
    return elements[index];
  }

  @Override
  public String toString() {
    return toStringHelper(this).add("elements", Arrays.asList(elements)).toString();
  }
}
