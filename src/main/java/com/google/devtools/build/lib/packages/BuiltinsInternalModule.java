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

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

// TODO(#11437): Note that if Stardoc's current design were to be long-lived, we'd want to factor
// out an API into starlarkbuildapi. As it is that almost certainly won't be necessary.
/**
 * The {@code _builtins} Starlark object, visible only to {@code @_builtins} .bzl files, supporting
 * access to internal APIs.
 */
@StarlarkBuiltin(
    name = "_builtins",
    category = DocCategory.BUILTIN,
    documented = false,
    doc =
        "A module accessible only to @_builtins .bzls, that permits access to the original "
            + "(uninjected) native builtins, as well as internal-only symbols not accessible to "
            + "users.")
public class BuiltinsInternalModule implements StarlarkValue {

  // _builtins.native
  private final Object uninjectedNativeObject;
  // _builtins.toplevel
  private final Object uninjectedToplevelObject;
  // _builtins.internal
  private final Object internalObject;

  public BuiltinsInternalModule(
      Object uninjectedNativeObject, Object uninjectedToplevelObject, Object internalObject) {
    this.uninjectedNativeObject = uninjectedNativeObject;
    this.uninjectedToplevelObject = uninjectedToplevelObject;
    this.internalObject = internalObject;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<_builtins module>");
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @StarlarkMethod(
      name = "native",
      doc =
          "A view of the <code>native</code> object as it would exist if builtins injection were"
              + " disabled. For example, if builtins injection provides a Starlark definition for"
              + " <code>cc_library</code> in <code>exported_rules</code>, then"
              + " <code>native.cc_library</code> in a user .bzl file would refer to that"
              + " definition, but <code>_builtins.native.cc_library</code> in a"
              + " <code>@_builtins</code> .bzl file would still be the one defined in Java code."
              + " (Note that for clarity and to avoid a conceptual cycle, the regular top-level"
              + " <code>native</code> object is not defined for <code>@_builtins</code> .bzl"
              + " files.)",
      documented = false,
      structField = true)
  public Object getUninjectedNativeObject() {
    return uninjectedNativeObject;
  }

  @StarlarkMethod(
      name = "toplevel",
      doc =
          "A view of the top-level .bzl symbols that would exist if builtins injection were"
              + " disabled; analogous to <code>_builtins.native</code>. For example, if builtins"
              + " injection provides a Starlark definition for <code>CcInfo</code> in"
              + " <code>exported_toplevels</code>, then <code>_builtins.toplevel.CcInfo</code>"
              + " refers to the original Java definition, not the Starlark one. (Just as for"
              + " <code>_builtins.native</code>, the top-level <code>CcInfo</code> symbol is not"
              + " available to <code>@_builtins</code> .bzl files.)",
      documented = false,
      structField = true)
  public Object getUninjectedToplevelObject() {
    return uninjectedToplevelObject;
  }

  @StarlarkMethod(
      name = "internal",
      doc =
          "A view of symbols that were registered (via the Java method"
              + "<code>ConfiguredRuleClassProvider#addStarlarkBuiltinsInternal</code>) to be made"
              + " available to <code>@_builtins</code> code but not necessarily user code.",
      documented = false,
      structField = true)
  public Object getInternalObject() {
    return internalObject;
  }

  @StarlarkMethod(
      name = "get_flag",
      doc =
          "Returns the value of a <code>StarlarkSemantics</code> flag, or a default value if it"
              + " could not be retrieved (either because the flag does not exist or because it was"
              + " not assigned an explicit value). Fails if the flag value exists but is not a"
              + " Starlark value.",
      documented = false,
      parameters = {
        @Param(name = "name", doc = "Name of the flag, without the leading dashes"),
        /*
         * Because of the way flag values are stored in StarlarkSemantics, we cannot retrieve a flag
         * whose value was not explicitly set, nor can we programmatically determine its default
         * value. This parameter is essentially a hack to avoid a slightly costlier refactoring of
         * StarlarkSemantics and BuildLanguageOptions. If the value passed in here by the caller
         * differs from the true default value of the flag, you could end up in a situation where
         * the semantics of some @_builtins code varies depending on whether a flag was set to its
         * default value implicitly or explicitly.
         */
        @Param(
            name = "default",
            doc =
                "Value to return if flag was not set or does not exist. This should always be set"
                    + " to the same value as the flag's default value.")
      },
      useStarlarkThread = true)
  public Object getFlag(String name, Object defaultValue, StarlarkThread thread) {
    Object value = thread.getSemantics().getGeneric(name, defaultValue);
    return Starlark.fromJava(value, thread.mutability());
  }
}
