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
