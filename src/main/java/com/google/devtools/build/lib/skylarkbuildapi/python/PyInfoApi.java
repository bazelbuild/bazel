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

package com.google.devtools.build.lib.skylarkbuildapi.python;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Provider instance for the Python rules. */
@SkylarkModule(
    name = "PyInfo",
    doc = "Encapsulates information provided by the Python rules.",
    category = SkylarkModuleCategory.PROVIDER)
public interface PyInfoApi<FileT extends FileApi> extends StarlarkValue {

  @SkylarkCallable(
      name = "transitive_sources",
      structField = true,
      doc =
          "A (<code>postorder</code>-compatible) depset of <code>.py</code> files appearing in the "
              + "target's <code>srcs</code> and the <code>srcs</code> of the target's transitive "
              + "<code>deps</code>.")
  Depset getTransitiveSources();

  @SkylarkCallable(
      name = "uses_shared_libraries",
      structField = true,
      doc =
          "Whether any of this target's transitive <code>deps</code> has a shared library file "
              + "(such as a <code>.so</code> file)."
              + ""
              + "<p>This field is currently unused in Bazel and may go away in the future.")
  boolean getUsesSharedLibraries();

  @SkylarkCallable(
      name = "imports",
      structField = true,
      doc =
          "A depset of import path strings to be added to the <code>PYTHONPATH</code> of "
              + "executable Python targets. These are accumulated from the transitive "
              + "<code>deps</code>."
              + ""
              + "<p>The order of the depset is not guaranteed and may be changed in the future. It "
              + "is recommended to use <code>default</code> order (the default).")
  Depset getImports();

  @SkylarkCallable(
      name = "has_py2_only_sources",
      structField = true,
      doc = "Whether any of this target's transitive sources requires a Python 2 runtime.")
  boolean getHasPy2OnlySources();

  @SkylarkCallable(
      name = "has_py3_only_sources",
      structField = true,
      doc = "Whether any of this target's transitive sources requires a Python 3 runtime.")
  boolean getHasPy3OnlySources();

  /** Provider type for {@link PyInfoApi} objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  interface PyInfoProviderApi extends ProviderApi {

    @SkylarkCallable(
        name = "PyInfo",
        doc = "The <code>PyInfo</code> constructor.",
        parameters = {
          @Param(
              name = "transitive_sources",
              type = Depset.class,
              generic1 = FileApi.class,
              positional = false,
              named = true,
              doc = "The value for the new object's <code>transitive_sources</code> field."),
          @Param(
              name = "uses_shared_libraries",
              type = Boolean.class,
              positional = false,
              named = true,
              defaultValue = "False",
              doc = "The value for the new object's <code>uses_shared_libraries</code> field."),
          @Param(
              name = "imports",
              type = Depset.class,
              generic1 = String.class,
              positional = false,
              named = true,
              defaultValue = "unbound",
              doc = "The value for the new object's <code>imports</code> field."),
          @Param(
              name = "has_py2_only_sources",
              type = Boolean.class,
              positional = false,
              named = true,
              defaultValue = "False",
              doc = "The value for the new object's <code>has_py2_only_sources</code> field."),
          @Param(
              name = "has_py3_only_sources",
              type = Boolean.class,
              positional = false,
              named = true,
              defaultValue = "False",
              doc = "The value for the new object's <code>has_py3_only_sources</code> field.")
        },
        selfCall = true,
        useStarlarkThread = true)
    @SkylarkConstructor(objectType = PyInfoApi.class, receiverNameForDoc = "PyInfo")
    PyInfoApi<?> constructor(
        Depset transitiveSources,
        boolean usesSharedLibraries,
        Object importsUncast,
        boolean hasPy2OnlySources,
        boolean hasPy3OnlySources,
        StarlarkThread thread)
        throws EvalException;
  }
}
