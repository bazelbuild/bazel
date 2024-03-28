// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options;

import javax.annotation.Nullable;

/**
 * A converter is a little helper object that can take a String and turn it into an instance of type
 * T (the type parameter to the converter). A context object is optionally provided.
 */
public interface Converter<T> {

  /**
   * Convert a string into type T, using the given conversion context. Please note that we assume
   * that converting the same string (if successful) will produce objects which are equal ({@link
   * Object#equals}).
   */
  T convert(String input, @Nullable Object conversionContext) throws OptionsParsingException;

  /**
   * The type description appears in usage messages. E.g.: "a string",
   * "a path", etc.
   */
  String getTypeDescription();

  /**
   * Can this converter reverse-convert to a Starlark-readable value?
   *
   * <p>If so, {@link #reverseForStarlark} implements the reverse conversion. If not, {@link
   * #reverseForStarlark} throws an {@link UnsupportedOperationException}.
   */
  default boolean starlarkConvertible() {
    return false;
  }

  /**
   * If {@link #starlarkConvertible()} is true, this reverses a converted value back to a
   * Starlark-readable form.
   *
   * <p>If {@link #starlarkConvertible()} is true, throws an {@link UnsupportedOperationException}.
   *
   * @param converted If the option this value represents isn't {@link Option#allowMultiple}, an
   *     object of the option's Java type. Else an entry in the option's {@link java.util.List}.
   *     Always of type T. Referenced as an {@link Object} because calling code can call any
   *     converter.
   * @return A {@link String} version of the input. Calling {@link #convert} on this value should
   *     faithfully reproduce {@code converted}.
   */
  default String reverseForStarlark(Object converted) {
    throw new UnsupportedOperationException("This converter doesn't support Starlark reversal.");
  }

  /** A converter that never reads its context parameter. */
  abstract class Contextless<T> implements Converter<T> {

    /**
     * Actual implementation of {@link #convert(String, Object)} that just ignores the context
     * parameter.
     */
    public abstract T convert(String input) throws OptionsParsingException;

    @Override
    public final T convert(String input, @Nullable Object conversionContext)
        throws OptionsParsingException {
      return convert(input);
    }
  }
}
