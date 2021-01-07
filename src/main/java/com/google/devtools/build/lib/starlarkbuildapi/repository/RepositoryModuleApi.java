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

package com.google.devtools.build.lib.starlarkbuildapi.repository;

import com.google.devtools.build.docgen.annot.DocumentMethods;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;

/**
 * The Starlark module containing the definition of {@code repository_rule} function to define a
 * Starlark remote repository.
 */
@DocumentMethods
public interface RepositoryModuleApi {

  @StarlarkMethod(
      name = "repository_rule",
      doc =
          "Creates a new repository rule. Store it in a global value, so that it can be loaded and "
              + "called from the WORKSPACE file.",
      parameters = {
        @Param(
            name = "implementation",
            named = true,
            doc =
                "the function that implements this rule. Must have a single parameter,"
                    + " <code><a href=\"repository_ctx.html\">repository_ctx</a></code>. The"
                    + " function is called during the loading phase for each instance of the"
                    + " rule."),
        @Param(
            name = "attrs",
            allowedTypes = {
              @ParamType(type = Dict.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            doc =
                "dictionary to declare all the attributes of the rule. It maps from an attribute "
                    + "name to an attribute object (see <a href=\"attr.html\">attr</a> "
                    + "module). Attributes starting with <code>_</code> are private, and can be "
                    + "used to add an implicit dependency on a label to a file (a repository "
                    + "rule cannot depend on a generated artifact). The attribute "
                    + "<code>name</code> is implicitly added and must not be specified.",
            named = true,
            positional = false),
        @Param(
            name = "local",
            defaultValue = "False",
            doc =
                "Indicate that this rule fetches everything from the local system and should be "
                    + "reevaluated at every fetch.",
            named = true,
            positional = false),
        @Param(
            name = "environ",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
            },
            defaultValue = "[]",
            doc =
                "Provides a list of environment variable that this repository rule depends on. If "
                    + "an environment variable in that list change, the repository will be "
                    + "refetched.",
            named = true,
            positional = false),
        @Param(
            name = "configure",
            defaultValue = "False",
            doc = "Indicate that the repository inspects the system for configuration purpose",
            named = true,
            positional = false),
        @Param(
            name = "remotable",
            defaultValue = "False",
            doc = "Compatible with remote execution",
            named = true,
            positional = false,
            enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_REPO_REMOTE_EXEC,
            valueWhenDisabled = "False"),
        @Param(
            name = "doc",
            defaultValue = "''",
            doc =
                "A description of the repository rule that can be extracted by documentation "
                    + "generating tools.",
            named = true,
            positional = false)
      },
      useStarlarkThread = true)
  StarlarkCallable repositoryRule(
      StarlarkCallable implementation,
      Object attrs,
      Boolean local,
      Sequence<?> environ, // <String> expected
      Boolean configure,
      Boolean remotable,
      String doc,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "__do_not_use_fail_with_incompatible_use_cc_configure_from_rules_cc",
      doc =
          "When --incompatible_use_cc_configure_from_rules_cc is set to true, Bazel will "
              + "fail the build. Please see https://github.com/bazelbuild/bazel/issues/10134 for "
              + "details and migration instructions.",
      documented = false,
      useStarlarkThread = true)
  void failWithIncompatibleUseCcConfigureFromRulesCc(StarlarkThread thread) throws EvalException;
}
