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
import com.google.devtools.build.android.desugar.langmodel.MethodKey;

/** The key to index a method in a inheritance hierarchy. */
@AutoValue
public abstract class HierarchicalMethodKey {

  public abstract HierarchicalTypeKey owner();

  public abstract HeadlessMethodKey headlessMethod();

  static HierarchicalMethodKey create(HierarchicalTypeKey owner, HeadlessMethodKey headlessMethod) {
    return new AutoValue_HierarchicalMethodKey(owner, headlessMethod);
  }

  public static HierarchicalMethodKey from(MethodKey methodKey) {
    return create(
        HierarchicalTypeKey.create(methodKey.owner()),
        HeadlessMethodKey.create(methodKey.name(), methodKey.descriptor()));
  }

  public final MethodKey toMethodKey() {
    return MethodKey.create(owner().type(), headlessMethod().name(), headlessMethod().descriptor());
  }

  /** Resolves the method in the given type hierarchy. */
  public HierarchicalMethodQuery inTypeHierarchy(TypeHierarchy typeHierarchy) {
    return typeHierarchy.query(this);
  }
}
