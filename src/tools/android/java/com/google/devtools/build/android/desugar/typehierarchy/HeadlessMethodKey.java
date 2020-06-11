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

/**
 * The key to index a method or constructor in the same class or interface. It is equivalent to
 * {@link MethodKey} with the exclusion of the declaration owner class, i.e. omitting the {@link
 * MethodKey#owner()} property.
 */
@AutoValue
abstract class HeadlessMethodKey {

  /** See: {@link MethodKey#name()}. */
  abstract String name();

  /** See: {@link MethodKey#descriptor()}. */
  abstract String descriptor();

  /** Factory method of {@link HeadlessMethodKey}. */
  static HeadlessMethodKey create(String methodName, String descriptor) {
    return new AutoValue_HeadlessMethodKey(methodName, descriptor);
  }
}
