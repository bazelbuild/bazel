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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;

/** Global constants and support for static registration of builtin symbols. */
// TODO(adonovan): move to Starlark.NONE and delete.
public final class Runtime {

  private Runtime() {}

  /** There should be only one instance of this type to allow "== None" tests. */
  @SkylarkModule(
    name = "NoneType",
    documented = false,
    doc = "Unit type, containing the unique value None."
  )
  @Immutable
  public static final class NoneType implements SkylarkValue {
    private NoneType() {}

    @Override
    public String toString() {
      return "None";
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Override
    public boolean truth() {
      return false;
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("None");
    }
  }

  /* The Starlark None value. */
  public static final NoneType NONE = new NoneType();
}
