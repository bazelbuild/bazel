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

package com.google.devtools.build.lib.skylarkbuildapi.repository;

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;

/**
 * The Skylark module containing the definition of {@code repository_rule} function to define a
 * skylark remote repository.
 */
@SkylarkGlobalLibrary
public interface RepositoryModuleApi {

  @SkylarkCallable(
      name = "repository_rule",
      doc =
          "Creates a new repository rule. Store it in a global value, so that it can be loaded and "
              + "called from the WORKSPACE file.",
      parameters = {
        @Param(
            name = "implementation",
            type = BaseFunction.class,
            legacyNamed = true,
            doc =
                "the function implementing this rule, has to have exactly one parameter: "
                    + "<code><a href=\"repository_ctx.html\">repository_ctx</a></code>. The function "
                    + "is called during loading phase for each instance of the rule."),
        @Param(
            name = "attrs",
            type = SkylarkDict.class,
            noneable = true,
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
            type = Boolean.class,
            defaultValue = "False",
            doc =
                "Indicate that this rule fetches everything from the local system and should be "
                    + "reevaluated at every fetch.",
            named = true,
            positional = false),
        @Param(
            name = "environ",
            type = SkylarkList.class,
            generic1 = String.class,
            defaultValue = "[]",
            doc =
                "Provides a list of environment variable that this repository rule depends on. If "
                    + "an environment variable in that list change, the repository will be "
                    + "refetched.",
            named = true,
            positional = false),
        @Param(
            name = "doc",
            type = String.class,
            defaultValue = "''",
            doc =
                "A description of the repository rule that can be extracted by documentation "
                    + "generating tools.",
            named = true,
            positional = false)
      },
      useAst = true,
      useEnvironment = true)
  public BaseFunction repositoryRule(
      BaseFunction implementation,
      Object attrs,
      Boolean local,
      SkylarkList<String> environ,
      String doc, 
      FuncallExpression ast,
      Environment env)
      throws EvalException;
}
