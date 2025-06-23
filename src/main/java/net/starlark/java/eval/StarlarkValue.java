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

package net.starlark.java.eval;

import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;

/** Base interface for all Starlark values besides boxed Java primitives. */
public interface StarlarkValue {
  default StarlarkType getStarlarkType() {
    return Types.ANY;
  }

  /**
   * Prints an official representation of object x.
   *
   * <p>Convention is that the string should be parseable back to the value x. If this isn't
   * feasible then it should be a short human-readable description enclosed in angled brackets, e.g.
   * {@code "<foo object>"}.
   *
   * @param printer a printer to be used for formatting nested values.
   * @deprecated use {@link #repr(Printer, StarlarkSemantics)} instead
   */
  @Deprecated
  default void repr(Printer printer) {
    repr(printer, StarlarkSemantics.DEFAULT);
  }

  /**
   * Prints an official representation of object x.
   *
   * <p>Convention is that the string should be parseable back to the value x. If this isn't
   * feasible then it should be a short human-readable description enclosed in angled brackets, e.g.
   * {@code "<foo object>"}.
   *
   * @param printer a printer to be used for formatting nested values.
   */
  default void repr(Printer printer, StarlarkSemantics semantics) {
    printer.append("<unknown object ").append(getClass().getName()).append(">");
  }

  /**
   * Prints an informal, human-readable representation of the value.
   *
   * <p>By default dispatches to the {@code repr} method.
   *
   * @param printer a printer to be used for formatting nested values.
   */
  default void str(Printer printer, StarlarkSemantics semantics) {
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
  default void debugPrint(Printer printer, StarlarkThread thread) {
    str(printer, thread.getSemantics());
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

  /**
   * Returns normally if the Starlark value is hashable and thus suitable as a dict key.
   *
   * <p>(A StarlarkValue implementation may define hashCode and equals and thus be a valid
   * java.util.Map key without being hashable by Starlark code.)
   *
   * @throws EvalException otherwise.
   */
  default void checkHashable() throws EvalException {
    // Bazel makes widespread assumptions that all Starlark values can be hashed
    // by Java code, so we cannot implement checkHashable by having
    // StarlarkValue.hashCode throw an unchecked exception, which would be more
    // efficient. Instead, before inserting a value in a dict, we must first check
    // whether it is hashable by calling this function, and then call its hashCode
    // method only if so.
    // For structs and tuples, this unfortunately visits the object graph twice.
    //
    // One subtlety: Bazel's lib.packages.StarlarkInfo.checkHashable, by using this
    // default implementation of checkHashable, which is based on isImmutable,
    // recursively asks whether its elements are immutable, not hashable.
    // Consequently, even though a list may not be used as a dict key (even if frozen),
    // a struct containing a list is hashable.
    // TODO(adonovan): fix this inconsistency. Requires a Bazel incompatible change.
    if (!this.isImmutable()) {
      throw Starlark.errorf("unhashable type: '%s'", Starlark.type(this));
    }
  }
}
