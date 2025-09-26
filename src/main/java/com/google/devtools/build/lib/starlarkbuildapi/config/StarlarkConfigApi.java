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

package com.google.devtools.build.lib.starlarkbuildapi.config;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.StarlarkValue;

/**
 * The "config" module of the Build API.
 *
 * <p>This exposes methods to describe what kind of build setting (if any) a starlark rule is using
 * the {@code build_setting} attr of the {@code rule(...)} function.
 */
@StarlarkBuiltin(
    name = "config",
    category = DocCategory.TOP_LEVEL_MODULE,
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
            defaultValue = "False",
            doc = FLAG_ARG_DOC,
            named = true,
            positional = false),
        @Param(
            name = "allow_multiple",
            defaultValue = "False",
            doc =
                "Deprecated, use a <code>string_list</code> setting with"
                    + " <code>repeatable = True</code> instead. If set, this flag is allowed to be"
                    + " set multiple times on the command line. The Value of the flag as accessed"
                    + " in transitions and build setting implementation function will be a list of"
                    + " strings. Insertion order and repeated values are both maintained. This list"
                    + " can be post-processed in the build setting implementation function if"
                    + " different behavior is desired.",
            named = true,
            positional = false)
      })
  BuildSettingApi stringSetting(Boolean flag, Boolean allowMultiple);

  @StarlarkMethod(
      name = "string_list",
      doc =
          "A string list-typed build setting. On the command line pass a list using"
              + " comma-separated value like <code>--//my/setting=foo,bar</code>.",
      parameters = {
        @Param(
            name = FLAG_ARG,
            defaultValue = "False",
            doc = FLAG_ARG_DOC,
            named = true,
            positional = false),
        @Param(
            name = "repeatable",
            defaultValue = "False",
            doc =
                "If set, instead of expecting a comma-separated value, this flag is allowed to be"
                    + " set multiple times on the command line with each individual value treated"
                    + " as a single string to add to the list value. Insertion order and repeated"
                    + " values are both maintained. This list can be post-processed in the build"
                    + " setting implementation function if different behavior is desired.",
            named = true,
            positional = false)
      })
  BuildSettingApi stringListSetting(Boolean flag, Boolean repeatable) throws EvalException;

  @StarlarkMethod(
      name = "string_set",
      doc =
          "A string set-typed build setting. The value of this setting will be a <a"
              + " href='https://bazel.build/rules/lib/core/set'>set</a> of strings in Starlark. On"
              + " the command line, pass a set using a comma-separated value like"
              + " <code>--//my/setting=foo,bar</code>."
              + "<p>Unlike with a <code>string_list</code>, the order of the elements doesn't"
              + " matter and only a single instance of each element is maintained. This is"
              + " recommended over <code>string_list</code> for flags where these properties are"
              + " not needed as it can improve build performance by avoiding unnecessary"
              + " configurations forking.",
      parameters = {
        @Param(
            name = FLAG_ARG,
            defaultValue = "False",
            doc = FLAG_ARG_DOC,
            named = true,
            positional = false),
        @Param(
            name = "repeatable",
            defaultValue = "False",
            doc =
                "If set, instead of expecting a comma-separated value, this flag is allowed to be"
                    + " set multiple times on the command line with each individual value treated"
                    + " as a single string to add to the set value. Only a single instance of"
                    + " repeated values is maintained and the insertion order does not matter.",
            named = true,
            positional = false)
      })
  BuildSettingApi stringSetSetting(Boolean flag, Boolean repeatable) throws EvalException;

  /** The API for build setting descriptors. */
  @StarlarkBuiltin(
      name = "BuildSetting",
      category = DocCategory.BUILTIN,
      doc =
          "The descriptor for a single piece of configuration information. If configuration is a "
              + "key-value map of settings like {'cpu': 'ppc', 'copt': '-DFoo'}, this describes a "
              + "single entry in that map.")
  interface BuildSettingApi extends StarlarkValue {}

  @StarlarkMethod(
      name = "exec",
      doc = "Creates an execution transition.",
      parameters = {
        @Param(
            name = "exec_group",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            defaultValue = "None",
            doc =
                "The name of the exec group whose execution platform this transition will use. If"
                    + " not provided, this exec transition will use the target's default execution"
                    + " platform.")
      })
  ExecTransitionFactoryApi exec(Object execGroupUnchecked);

  @StarlarkMethod(
      name = "target",
      doc =
          "Creates a target transition. This is a no-op transition intended for the case where a"
              + " transition object is needed, but doesn't want to actually change anything."
              + " Equivalent to <code>cfg = \"target\"</code> in <code>attr.label()</code>.")
  ConfigurationTransitionApi target();

  @StarlarkMethod(
      name = "none",
      doc =
          "Creates a transition which removes all configuration, unsetting all flags. Intended for"
              + " the case where a dependency is data-only and contains no code that needs to be"
              + " built, but should only be analyzed once.")
  ConfigurationTransitionApi none();

  /** The api for exec transitions. */
  @StarlarkBuiltin(
      name = "ExecTransitionFactory",
      category = DocCategory.BUILTIN,
      doc = "an execution transition.")
  interface ExecTransitionFactoryApi extends ConfigurationTransitionApi {}
}
