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

package com.google.devtools.build.android.desugar.langmodel;

/**
 * Imposes that the implementing class is to support deep type remapping with a given {@link
 * TypeMapper}.
 */
@FunctionalInterface
public interface TypeMappable<T> {

  /**
   * Accepts a type mapper and returns a new instance of remapped struct without changing the
   * original source instance. Please apply {@param typeMapper} to any index-able state of the
   * implementation class.
   */
  T acceptTypeMapper(TypeMapper typeMapper);
}
