// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.Location;

/**
 * An Info is a unit of information produced by analysis of one configured target and consumed by
 * other targets that depend directly upon it. The result of analysis is a dictionary of Info
 * values, each keyed by its Provider. Every Info is an instance of a Provider: if a Provider is
 * like a Java class, then an Info is like an instance of that class.
 */
// TODO(adonovan): simplify the hierarchies below in these steps:
// - Once to_{json,proto} are gone, StructApi can be deleted; structs should never again have
//   methods.
// - StructImpl.location can be pushed down into subclasses that need it, much as we did for
//   StructImpl.provider in this CL.
// - getErrorMessageFormatForUnknownField can become a method on provider.
//   It should compute a string from a parameter, not use higher-order formatting.
// - StructImpl is then really just a collection of helper functions for subclasses
//   getValue(String, Class), repr, equals, hash. Move them, and merge it into Info interface.
// - Move StructProvider.STRUCT and make StructProvider private.
//   The StructProvider.createStruct method could be a simple function like depset, select.
//   StructProviderApi could be eliminated.
// - eliminate StarlarkInfo + StarlarkInfo.
// - NativeInfo's two methods can (IIUC) be deleted immediately, and then NativeInfo itself.
//
// Info (result of analysis)
// - StructImpl (structure with fields, to_{json,proto}). Implements Structure, StructApi.
//   - OutputGroupInfo. Fields are output group names.
//   - NativeInfo. Fields are Java annotated methods (tricky).
//     - dozens of subclasses
//   - StarlarkInfo. Has table of k/v pairs. Final. Supports x+y.
//
// Provider (key for analysis result Info; class symbol for StructImpls). Implements ProviderApi.
// - BuiltinProvider
//   - StructProvider (for basic 'struct' values). Callable. Implements ProviderApi.
//   - dozens of singleton subclasses
// - StarlarkProvider. Callable.
//
public interface Info extends StarlarkValue {

  /** Returns the provider that instantiated this Info. */
  Provider getProvider();

  /**
   * Returns the source location where this Info (provider instance) was created, or BUILTIN if it
   * was instantiated by Java code.
   */
  default Location getCreationLocation() {
    return Location.BUILTIN;
  }

  @Override
  default void repr(Printer printer) {
    printer.append("<instance of provider ");
    printer.append(getProvider().getPrintableName());
    printer.append(">");
  }
}
