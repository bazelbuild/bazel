// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import java.util.function.Consumer;

/** An interface for an object that customizes how it is expanded into a command line. */
public interface CommandLineItem {
  /**
   * A map function that allows caller customization how a type is expanded into the command line.
   */
  interface MapFn<T> {
    MapFn<Object> DEFAULT =
        (Object object, Consumer<String> args) ->
            args.accept(CommandLineItem.expandToCommandLine(object));

    void expandToCommandLine(T object, Consumer<String> args);
  }

  /**
   * Use this map function when parametrizing over a limited set of values.
   *
   * <p>The user promises that the number of distinct instances constructed is closer to O(rule
   * class count) than O(rule count).
   *
   * <p>Without this, {@link
   * com.google.devtools.build.lib.collect.nestedset.NestedSetFingerprintCache} will refuse to cache
   * your {@link MapFn} computations.
   */
  abstract class ParametrizedMapFn<T> implements MapFn<T> {
    @Override
    public abstract boolean equals(Object obj);

    @Override
    public abstract int hashCode();

    /**
     * This method controls the max number of distinct instances allowed. If the system sees any
     * more than this, it will throw.
     *
     * <p>Override and set this to something low. You want this to represent the small number of
     * preallocated static instances used in this blaze instance. 3 is an OK number, 100 is a bad
     * number.
     */
    public abstract int maxInstancesAllowed();
  }

  /**
   * Use this map function when your map function needs to capture per-rule information.
   *
   * <p>Use of this class prevents sharing sub-computations over shared NestedSets, since the map
   * function is per-target. This will make your action key computations become O(N^2). Please avoid
   * if possible.
   */
  interface CapturingMapFn<T> extends MapFn<T> {}

  /** Expands the object into the command line as a string. */
  String expandToCommandLine();

  /**
   * The default method of expanding types.
   *
   * <p>If the object is a {@link CommandLineItem} we use its {@link
   * CommandLineItem#expandToCommandLine} method, else we call {@link Object#toString()}.
   */
  static String expandToCommandLine(Object object) {
    // TODO(b/150322434): The fallback on toString() isn't great. Particularly so for
    // SkylarkCustomCommandLine, since toString() does not necessarily give the same results as
    // Skylark's str() or repr().
    //
    // The ideal refactoring is to make StarlarkValue implement CommandLineItem (or a slimmer
    // version
    // thereof). Then the default behavior can be that StarlarkValue#expandToCommandLine calls
    // StarlarkValue#str. This default behavior is inefficient but rare; Artifacts and the like
    // would continue to override expandToCommandLine to take the fast code path that doesn't
    // involve a Printer.
    //
    // Since StarlarkValue should be moved out of Bazel, this refactoring would be blocked on making
    // a BuildStarlarkValue subinterface for Bazel-specific Skylark types. It would then be
    // BuildStarlarkValue, rather than StarlarkValue, that extends CommandLineItem.
    if (object instanceof CommandLineItem) {
      return ((CommandLineItem) object).expandToCommandLine();
    } else {
      return object.toString();
    }
  }
}
