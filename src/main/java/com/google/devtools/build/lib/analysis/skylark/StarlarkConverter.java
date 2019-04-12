// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.skylark;

import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.common.options.Converter;

/**
 * A converter that also knows how to turn a native object into a Starlark object
 *
 * <p>This interface declares/inherits two converting methods:
 *
 * <p>For converting from command line to Java (inherited)
 *
 * <p>{@code convert} String -> Object<T>
 *
 * <p>For converting from Java to Starlark
 *
 * <p>{@code convertToStarlark} Object<S> -> SkylarkValue
 *
 * <p>In practice, <T> and <S> will almost always be the same type. An example of an exception to
 * this rule is for options with {@code allowMultiple = true} for which <S> is a list of <T>
 * objects.
 */
// TODO(juliexxia): Add Starlark -> Java converter method to this interface.
public interface StarlarkConverter<S, T> extends Converter<T> {

  /** Convert a java object of type S into a Starlark friendly value. */
  // TODO(bazel-team): exapnd this interface to allow "primitive" types to also be returned here.
  // Example case of where this would be useful is for {@code enum}s.
  SkylarkValue convertToStarlark(S input);
}
