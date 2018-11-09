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

package com.google.devtools.build.lib.skylarkbuildapi.config;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import java.util.List;

/**
 * A collection of top-level Starlark functions pertaining to configuration.
 */
@SkylarkGlobalLibrary
public interface ConfigGlobalLibraryApi {
  @SkylarkCallable(
      name = "transition",
      // TODO(cparsons): Improve documentation with an example once this feature is
      // non-experimental.
      doc =
          "<b>Experimental. This type is experimental and subject to change at any time. Do "
              + "not depend on it.</b><p> Creates a configuration transition to be applied across"
              + " a dependency edge.",
      parameters = {
        @Param(
            name = "implementation",
            type = BaseFunction.class,
            positional = false,
            named = true,
            // TODO(cparsons): The settings dict should take actual Label objects as keys and not
            // strings. Update the documentation.
            doc =
                "The function implementing this transition. This function always has the "
                    + "parameter <code>settings</code>, a dictionary whose set of keys is defined "
                    + "by the inputs parameter. So, for each build setting "
                    + "<code>--//foo=bar</code>, if <code>inputs</code> contains "
                    + "<code>//foo</code>, <code>settings</code> will "
                    + "have an entry <code>settings['//foo']='bar'</code>.<p>"
                    // TODO(cparsons): Consider making this parameter mandatory, and determine
                    // what to do with attributes which are defined with select().
                    + "This function also optionally takes a parameter <code>attr</code> which is "
                    + "a reference to <code>ctx.attr</code> but pre-analysis-phase. This gives the "
                    + "implementation function access to the rule's attributes to make "
                    + "attribute-parameterized transitions possible.<p>"
                    // TODO(cparsons): Mention the expected output for split transitions.
                    + "This function must return a <code>dict</code> from build setting identifier "
                    + "to build setting value; this represents the configuration transition: for "
                    + "each entry in the returned <code>dict</code>, the transition updates that "
                    + "setting to the new value. All other settings are unchanged."),
        @Param(
            name = "inputs",
            type = SkylarkList.class,
            generic1 = String.class,
            positional = false,
            named = true,
            doc =
                "List of build settings that can be read by this transition. This becomes the "
                    + "key set of the settings parameter of the implementation function "
                    + "parameter."),
        @Param(
            name = "outputs",
            type = SkylarkList.class,
            generic1 = String.class,
            positional = false,
            named = true,
            doc =
                "List of build settings that can be written by this transition. This must be "
                    + "a superset of the key set of the dictionary returned by this transition."),
      },
      useLocation = true,
      useEnvironment = true)
  @SkylarkConstructor(objectType = ConfigurationTransitionApi.class)
  public ConfigurationTransitionApi transition(
      BaseFunction implementation,
      List<String> inputs,
      List<String> outputs,
      Location location,
      Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "analysis_test_transition",
      // TODO(cparsons): Improve documentation with an example once this feature is
      // non-experimental.
      doc =
          "<b>Experimental. This type is experimental and subject to change at any time. Do "
              + "not depend on it.</b><p> Creates a configuration transition to be applied on "
              + "an analysis-test rule's dependencies. This transition may only be applied "
              + "on attributes of rules with <code>analysis_test = True</code>.",
      parameters = {
        @Param(
            name = "settings",
            type = SkylarkDict.class,
            positional = false,
            named = true,
            doc =
                "A dictionary containing information about configuration settings which "
                    + "should be set by this configuration transition. Keys are build setting "
                    + "labels and values are their new post-transition values. All other settings "
                    + "are unchanged. Use this to declare specific configuration settings that "
                    + "an analysis test requires to be set in order to pass."),
      },
      useLocation = true,
      useSkylarkSemantics = true)
  public ConfigurationTransitionApi analysisTestTransition(
      SkylarkDict<String, String> changedSettings, Location location, SkylarkSemantics semantics)
      throws EvalException;
}
