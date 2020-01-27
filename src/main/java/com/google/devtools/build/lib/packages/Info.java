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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/**
 * An Info is a unit of information produced by analysis of one configured target and consumed by
 * other targets that depend directly upon it. The result of analysis is a dictionary of Info
 * values, each keyed by its Provider. Every Info is an instance of a Provider: if a Provider is
 * like a Java class, then an Info is like an instance of that class.
 */
public interface Info extends StarlarkValue {

  /** Returns the provider that instantiated this Info. */
  Provider getProvider();

  /**
   * Returns the source location where this Info (provider instance) was created, or BUILTIN if it
   * was instantiated by Java code.
   */
  default Location getCreationLoc() {
    return Location.BUILTIN;
  }

  /**
   * This method (which is redundant with getCreationLoc and should not be overridden or called) is
   * required to pacify the AutoCodec annotation processor.
   */
  // TODO(adonovan): find out why and stop it.
  // Alternatively rename various constructor parameters from 'location' to 'creationLoc'.
  default Location getLocation() {
    return getCreationLoc();
  }

  @Override
  default void repr(Printer printer) {
    printer.append("<instance of provider ");
    printer.append(getProvider().getPrintableName());
    printer.append(">");
  }
}
