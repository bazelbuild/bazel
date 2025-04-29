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

package net.starlark.java.eval;

import javax.annotation.concurrent.Immutable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;

/** The type of the Starlark None value. */
@StarlarkBuiltin(
    name = "NoneType",
    documented = false,
    doc = "The type of the Starlark None value.")
@Immutable
public final class NoneType implements StarlarkValue {
  @Override
  public StarlarkType getStarlarkType() {
    return Types.NONE;
  }

  static final NoneType NONE = new NoneType();

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
  public void repr(Printer printer) {
    printer.append("None");
  }
}
