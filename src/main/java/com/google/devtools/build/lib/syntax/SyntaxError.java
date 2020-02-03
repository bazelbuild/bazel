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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import java.util.List;

/**
 * An exception that indicates a static error associated with the syntax, such as scanner or parse
 * error, a structural problem, or a failure of identifier resolution. The exception records one or
 * more errors, each with a syntax location.
 *
 * <p>SyntaxError is thrown by operations such as {@link Expression#parse}, which are "all or
 * nothing". By contrast, {@link StarlarkFile#parse} does not throw an exception; instead, it
 * records the accumulated scanner, parser, and optionally validation errors within the syntax tree,
 * so that clients may obtain partial information from a damaged file.
 *
 * <p>Clients that fail abruptly when encountering parse errors are encouraged to use SyntaxError,
 * as in this example:
 *
 * <pre>
 * StarlarkFile file = StarlarkFile.parse(input);
 * if (!file.ok()) {
 *     throw new SyntaxError(file.errors());
 * }
 * </pre>
 */
public final class SyntaxError extends Exception {

  private final ImmutableList<Event> errors;

  /** Construct a SyntaxError from a non-empty list of errors. */
  public SyntaxError(List<Event> errors) {
    if (errors.isEmpty()) {
      throw new IllegalArgumentException("no errors");
    }
    this.errors = ImmutableList.copyOf(errors);
  }

  /** Returns an immutable non-empty list of errors. */
  public ImmutableList<Event> errors() {
    return errors;
  }

  @Override
  public String getMessage() {
    String first = errors.get(0).getMessage();
    if (errors.size() > 1) {
      return String.format("%s (+ %d more)", first, errors.size() - 1);
    } else {
      return first;
    }
  }
}
