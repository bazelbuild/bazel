// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlarkbuildapi.java;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaPluginInfoApi.JavaPluginDataApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Info object encapsulating information about Java plugins. */
@StarlarkBuiltin(
    name = "JavaPluginInfo",
    doc =
        "A provider encapsulating information about Java plugins. "
            + "<p>At the moment, the only supported kind of plugins are annotation processors.",
    category = DocCategory.PROVIDER)
public interface JavaPluginInfoApi<JavaPluginDataT extends JavaPluginDataApi> extends StructApi {
  @StarlarkMethod(name = "plugins", doc = "Returns data about all plugins.", structField = true)
  JavaPluginDataT plugins();

  @StarlarkMethod(
      name = "api_generating_plugins",
      doc =
          "Returns data about API generating plugins. "
              + "<p>Those annotation processors are applied to a Java target before producing "
              + "its header jars (which contain method signatures). When no API plugins are "
              + "present, header jars are generated from the sources, reducing critical path. "
              + "<p>The <code>api_generating_plugins</code> is a subset of <code>plugins</code>.",
      structField = true)
  JavaPluginDataT apiGeneratingPlugins();

  /** Info object encapsulating information about a Java compatible plugin. */
  @StarlarkBuiltin(
      name = "JavaPluginData",
      category = DocCategory.PROVIDER,
      doc =
          "Information about a Java compatible plugin."
              + "<p>That is an annotation processor recognized by the Java compiler.")
  interface JavaPluginDataApi extends StarlarkValue {
    @StarlarkMethod(
        name = "processor_jars",
        doc = "Returns the jars needed to apply the encapsulated annotation processors.",
        structField = true)
    Depset /*<FileApi>*/ getProcessorJarsForStarlark();

    @StarlarkMethod(
        name = "processor_classes",
        doc =
            "Returns the fully qualified class names needed to apply the encapsulated annotation"
                + " processors.",
        structField = true)
    Depset /*<String>*/ getProcessorClassesForStarlark();

    @StarlarkMethod(
        name = "processor_data",
        doc =
            "Returns the files needed during execution by the encapsulated annotation processors.",
        structField = true)
    Depset /*<FileApi>*/ getProcessorDataForStarlark();
  }
}
