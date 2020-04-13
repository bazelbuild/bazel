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

package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.util.List;

/**
 * A SyntaxError represents a static error associated with the syntax, such as a scanner or parse
 * error, a structural problem, or a failure of identifier resolution. It records a description of
 * the error and its location in the syntax.
 */
public final class SyntaxError {

  private final Location location;
  private final String message;

  public SyntaxError(Location location, String message) {
    this.location = Preconditions.checkNotNull(location);
    this.message = Preconditions.checkNotNull(message);
  }

  /** Returns the location of the error. */
  public Location location() {
    return location;
  }

  /** Returns a description of the error. */
  public String message() {
    return message;
  }

  /** Returns a string of the form {@code "foo.star:1:2: oops"}. */
  @Override
  public String toString() {
    return location + ": " + message;
  }

  /**
   * A SyntaxError.Exception is an exception holding one or more syntax errors.
   *
   * <p>SyntaxError.Exception is thrown by operations such as {@link Expression#parse}, which are
   * "all or nothing". By contrast, {@link StarlarkFile#parse} does not throw an exception; instead,
   * it records the accumulated scanner, parser, and optionally validation errors within the syntax
   * tree, so that clients may obtain partial information from a damaged file.
   *
   * <p>Clients that fail abruptly when encountering parse errors are encouraged to throw
   * SyntaxError.Exception, as in this example:
   *
   * <pre>
   * StarlarkFile file = StarlarkFile.parse(input);
   * if (!file.ok()) {
   *     throw new SyntaxError.Exception(file.errors());
   * }
   * </pre>
   */
  public static final class Exception extends java.lang.Exception {

    private final ImmutableList<SyntaxError> errors;

    /** Construct a SyntaxError from a non-empty list of errors. */
    public Exception(List<SyntaxError> errors) {
      if (errors.isEmpty()) {
        throw new IllegalArgumentException("no errors");
      }
      this.errors = ImmutableList.copyOf(errors);
    }

    /** Returns an immutable non-empty list of errors. */
    public ImmutableList<SyntaxError> errors() {
      return errors;
    }

    @Override
    public String getMessage() {
      String first = errors.get(0).message();
      if (errors.size() > 1) {
        // TODO(adonovan): say ("+ n more errors") to avoid ambiguity.
        return String.format("%s (+ %d more)", first, errors.size() - 1);
      } else {
        return first;
      }
    }
  }
}
