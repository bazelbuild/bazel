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
package com.google.devtools.build.android;

import com.android.resources.ResourceType;
import com.google.devtools.build.android.resources.Visibility;
import java.io.Flushable;
import java.util.Map;

/** Defines a sink for collecting data about resource symbols. */
public abstract class AndroidResourceSymbolSink implements Flushable {

  public final void acceptSimpleResource(
      DependencyInfo dependencyInfo, Visibility visibility, ResourceType type, String name) {
    if (isPrivateResourceFromDependency(dependencyInfo, visibility)) {
      return;
    }
    acceptSimpleResourceImpl(dependencyInfo, visibility, type, name);
  }

  abstract void acceptSimpleResourceImpl(
      DependencyInfo dependencyInfo, Visibility visibility, ResourceType type, String name);

  // "inlineable" below affects how resource IDs are assigned by
  // PlaceholderIdFieldInitializerBuilder to attempt to match the final IDs assigned by aapt1.  This
  // shouldn't matter, but legacy tests with ODR violations might be relying on this.
  public final void acceptStyleableResource(
      DependencyInfo dependencyInfo,
      Visibility visibility,
      FullyQualifiedName key,
      Map<FullyQualifiedName, /*inlineable=*/ Boolean> attrs) {
    if (isPrivateResourceFromDependency(dependencyInfo, visibility)) {
      return;
    }
    acceptStyleableResourceImpl(dependencyInfo, visibility, key, attrs);
  }

  abstract void acceptStyleableResourceImpl(
      DependencyInfo dependencyInfo,
      Visibility visibility,
      FullyQualifiedName key,
      Map<FullyQualifiedName, /*inlineable=*/ Boolean> attrs);

  private static boolean isPrivateResourceFromDependency(
      DependencyInfo dependencyInfo, Visibility visibility) {
    return visibility == Visibility.PRIVATE
        && dependencyInfo.dependencyType() != DependencyInfo.DependencyType.PRIMARY
        && dependencyInfo.dependencyType() != DependencyInfo.DependencyType.UNKNOWN;
  }
}
