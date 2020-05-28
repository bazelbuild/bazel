// Copyright 2015 The Bazel Authors. All rights reserved.
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

/** Base interface for all Starlark values besides boxed Java primitives. */
public interface StarlarkValue {

  /**
   * Prints an official representation of object x.
   *
   * <p>Convention is that the string should be parseable back to the value x. If this isn't
   * feasible then it should be a short human-readable description enclosed in angled brackets, e.g.
   * {@code "<foo object>"}.
   *
   * @param printer a printer to be used for formatting nested values.
   */
  default void repr(Printer printer) {
    printer.append("<unknown object ").append(getClass().getName()).append(">");
  }

  /**
   * Prints an informal, human-readable representation of the value.
   *
   * <p>By default dispatches to the {@code repr} method.
   *
   * @param printer a printer to be used for formatting nested values.
   */
  default void str(Printer printer) {
    repr(printer);
  }

  /**
   * Prints an informal debug representation of the value.
   *
   * <p>This debug representation is only ever printed to the terminal or to another out-of-band
   * channel, and is never accessible to Starlark code. Therefore, it is safe for the debug
   * representation to reveal properties of the value that are usually hidden for the sake of
   * performance, determinism, or forward-compatibility.
   *
   * <p>By default dispatches to the {@code str} method.
   *
   * @param printer a printer to be used for formatting nested values.
   */
  default void debugPrint(Printer printer) {
    str(printer);
  }

  /** Returns the truth-value of this Starlark value. */
  default boolean truth() {
    return true;
  }

  /** Reports whether the value is deeply immutable. */
  // TODO(adonovan): eliminate this concept. All uses really need to know is, is it hashable?,
  // because Starlark values must have stable hashes: a hashable value must either be immutable or
  // its hash must be part of its identity.
  // But this must wait until --incompatible_disallow_hashing_frozen_mutables=true is removed.
  // (see github.com/bazelbuild/bazel/issues/7800)
  default boolean isImmutable() {
    return false;
  }

  /** Reports whether the Starlark value is hashable and thus suitable as a dict key. */
  default boolean isHashable() {
    return this.isImmutable();
  }
}
