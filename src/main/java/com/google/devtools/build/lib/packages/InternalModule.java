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

import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;

// TODO(#11437): Factor an API out into skylarkbuildapi, for stardoc's benefit. Otherwise, stardoc
// can't run on @builtins bzls.
/** The {@code _internal} Starlark object, visible only to {@code @builtins} .bzls. */
@StarlarkBuiltin(
    name = "_internal",
    category = StarlarkDocumentationCategory.BUILTIN,
    documented = false,
    doc =
        "A module accessible only to @builtins .bzls, that permits access to the original "
            + "(uninjected) native builtins, as well as internal-only symbols not accessible to "
            + "users.")
public class InternalModule implements StarlarkValue {

  // TODO(#11437): Can't use a singleton once we're re-exporting fields of the native object.
  public static final InternalModule INSTANCE = new InternalModule();

  private InternalModule() {}

  @Override
  public void repr(Printer printer) {
    printer.append("<_internal module>");
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public boolean isHashable() {
    return true;
  }
}
