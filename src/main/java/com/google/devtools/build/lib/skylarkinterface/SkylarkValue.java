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

package com.google.devtools.build.lib.skylarkinterface;

/**
 * Java objects that are also Skylark values.
 *
 * <p>This is used for extending the Skylark interpreter with domain-specific values.
 */
public interface SkylarkValue {

  /**
   * Returns if the value is immutable and thus suitable for being used as a dictionary key.
   *
   * <p>Immutability is deep, i.e. in order for a value to be immutable, all values it is composed
   * of must be immutable, too.
   */
  default boolean isImmutable() {
      return false;
  }

  /**
   * Print an official representation of object x.
   *
   * <p>For regular data structures, the value should be parsable back into an equal data structure.
   *
   * @param printer a printer to be used for formatting nested values.
   */
  void repr(SkylarkPrinter printer);

  /**
   * Print a legacy representation of object x.
   *
   * <p>By default dispatches to the {@code repr} method. Should be called instead of {@code repr}
   * if --incompatible_descriptive_string_representations=false is used.
   *
   * @param printer an instance of a printer to be used for formatting nested values
   */
  default void reprLegacy(SkylarkPrinter printer) {
    repr(printer);
  }

  /**
   * Print an informal, human-readable representation of the value.
   *
   * <p>By default dispatches to the {@code repr} method.
   *
   * @param printer a printer to be used for formatting nested values.
   */
  default void str(SkylarkPrinter printer) {
    repr(printer);
  }

  /**
   * Print a legacy informal, human-readable representation of the value.
   *
   * <p>By default dispatches to the {@code reprLegacy} method.
   *
   * @param printer a printer to be used for formatting nested values.
   */
  default void strLegacy(SkylarkPrinter printer) {
    reprLegacy(printer);
  }
}
