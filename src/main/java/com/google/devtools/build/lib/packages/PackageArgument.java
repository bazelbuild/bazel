// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Location;

/** Defines an argument to the {@code package()} function. */
public abstract class PackageArgument<T> {
  private final String name;
  private final Type<T> type;

  protected PackageArgument(String name, Type<T> type) {
    this.name = name;
    this.type = type;
  }

  String getName() {
    return name;
  }

  /**
   * Converts an untyped argument to a typed one, then calls the user-provided {@link #process}.
   *
   * Note that the location is used not just for exceptions (for which null would do), but also for
   * reporting events.
   */
  final void convertAndProcess(
      Package.Builder pkgBuilder, Location location, Object value)
      throws EvalException {
    T typedValue = type.convert(value, "'package' argument", pkgBuilder.getBuildFileLabel());
    process(pkgBuilder, location, typedValue);
  }

  /**
   * Processes an argument.
   *
   * @param pkgBuilder the package builder to be mutated
   * @param location the location of the {@code package} function for error reporting
   * @param value the value of the argument. Typically passed to {@link Type#convert}
   */
  protected abstract void process(
      Package.Builder pkgBuilder, Location location, T value)
      throws EvalException;
}
