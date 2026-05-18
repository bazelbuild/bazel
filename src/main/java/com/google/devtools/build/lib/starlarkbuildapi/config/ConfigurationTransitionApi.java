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
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

/** Represents a configuration transition across a dependency edge. */
@StarlarkBuiltin(
    name = "transition",
    category = DocCategory.BUILTIN,
    doc =
        "<p>Represents a configuration transition across a dependency edge. For example, if"
            + " <code>//package:foo</code> depends on <code>//package:bar</code> with a"
            + " configuration transition, then the configuration of <code>//package:bar</code> (and"
            + " its dependencies) will be <code>//package:foo</code>'s configuration plus the"
            + " changes specified by the transition function.")
public interface ConfigurationTransitionApi extends StarlarkValue {

  @StarlarkMethod(
      name = "and_then",
      doc =
          "Returns a new transition that applies this transition followed by the given one. The"
              + " second transition reads the build settings produced by this one; the original"
              + " transitions are left unchanged. The result is itself a transition and may be"
              + " composed further.<p>A composition may be used as a rule or attribute transition"
              + " wherever its component transitions could be used. At most one of the composed"
              + " transitions may target the exec configuration (e.g. <code>config.exec</code>)."
              + " When two transitions in the chain split the configuration, the result has the"
              + " cross product of their splits; the key for each combined split is the"
              + " comma-separated concatenation of the component keys.",
      parameters = {@Param(name = "transition", doc = "The transition to apply after this one.")})
  default ConfigurationTransitionApi andThen(ConfigurationTransitionApi transition)
      throws EvalException {
    return ComposedConfigurationTransitionApi.compose(this, transition);
  }
}
