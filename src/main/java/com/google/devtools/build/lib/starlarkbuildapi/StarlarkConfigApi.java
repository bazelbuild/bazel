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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/**
 * The "config" module of the Build API.
 *
 * <p>This exposes methods to describe what kind of build setting (if any) a starlark rule is using
 * the {@code build_setting} attr of the {@code rule(...)} function.
 */
@StarlarkBuiltin(
    name = "config",
    category = StarlarkDocumentationCategory.BUILTIN,
    doc =
        "This is a top-level module for creating configuration transitions and build "
            + "setting descriptors which describe what kind of build setting (if any) a rule is. "
            + ""
            + "<p>ex: the following rule is marked as a build setting by setting the "
            + "<code>build_setting</code> parameter of the <code>rule()</code> function. "
            + "Specifically it is a build setting of type <code>int</code> and is a "
            + "<code>flag</code> which means "
            + "this build setting is callable on the command line.<br><pre class=language-python>"
            + "  my_rule = rule(\n"
            + "    implementation = _impl,\n"
            + "    build_setting = config.int(flag = True),\n"
            + "    ...\n"
            + "  )</pre>")
public interface StarlarkConfigApi extends StarlarkValue {

  String FLAG_ARG = "flag";
  String FLAG_ARG_DOC = "Whether or not this build setting is callable on the command line.";

  @StarlarkMethod(
      name = "int",
      doc = "An integer-typed build setting",
      parameters = {
        @Param(
            name = FLAG_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = FLAG_ARG_DOC,
            named = true,
            positional = false)
      })
  BuildSettingApi intSetting(Boolean flag);

  @StarlarkMethod(
      name = "bool",
      doc = "A bool-typed build setting",
      parameters = {
        @Param(
            name = FLAG_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = FLAG_ARG_DOC,
            named = true,
            positional = false)
      })
  BuildSettingApi boolSetting(Boolean flag);

  @StarlarkMethod(
      name = "string",
      doc = "A string-typed build setting",
      parameters = {
        @Param(
            name = FLAG_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = FLAG_ARG_DOC,
            named = true,
            positional = false)
      })
  BuildSettingApi stringSetting(Boolean flag);

  @StarlarkMethod(
      name = "string_list",
      doc = "A string list-typed build setting",
      parameters = {
        @Param(
            name = FLAG_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = FLAG_ARG_DOC,
            named = true,
            positional = false)
      })
  BuildSettingApi stringListSetting(Boolean flag);

  /** The API for build setting descriptors. */
  @StarlarkBuiltin(
      name = "BuildSetting",
      category = StarlarkDocumentationCategory.BUILTIN,
      doc =
          "The descriptor for a single piece of configuration information. If configuration is a "
              + "key-value map of settings like {'cpu': 'ppc', 'copt': '-DFoo'}, this describes a "
              + "single entry in that map.")
  interface BuildSettingApi extends StarlarkValue {}

  @StarlarkMethod(
      name = "exec",
      doc = "<i>experimental</i> Creates an execution transition.",
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_EXEC_GROUPS,
      parameters = {
        @Param(
            name = "exec_group",
            type = String.class,
            named = true,
            noneable = true,
            defaultValue = "None",
            doc =
                "The name of the exec group whose execution platform this transition will use. If"
                    + " not provided, this exec transition will use the target's default execution"
                    + " platform.")
      })
  ExecTransitionFactoryApi exec(Object execGroupUnchecked);

  /** The api for exec transitions. */
  @StarlarkBuiltin(
      name = "ExecTransitionFactory",
      category = StarlarkDocumentationCategory.BUILTIN,
      doc = "<i>experimental</i> an execution transition.")
  interface ExecTransitionFactoryApi extends StarlarkValue {}
}
