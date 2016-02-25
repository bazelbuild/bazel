// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.skylark;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.syntax.SkylarkType.castMap;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.rules.SkylarkAttr.Descriptor;
import com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature.Param;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;

import java.util.Map;

/**
 * The Skylark module containing the definition of {@code repository_rule} function to define a
 * skylark remote repository.
 */
public class SkylarkRepositoryModule {

  @SkylarkSignature(
    name = "repository_rule",
    doc =
        "Creates a new repository rule. Store it in a global value, so that it can be loaded and "
            + "called from the WORKSPACE file.",
    returnType = BaseFunction.class,
    mandatoryPositionals = {
      @Param(
        name = "implementation",
        type = BaseFunction.class,
        doc =
            "the function implementing this rule, has to have exactly one parameter: "
                + "<code>ctx</code>. The function is called during analysis phase for each "
                + "instance of the rule."
      )
    },
    optionalNamedOnly = {
      @Param(
        name = "attrs",
        type = SkylarkDict.class,
        noneable = true,
        defaultValue = "None",
        doc =
            "dictionary to declare all the attributes of the rule. It maps from an attribute "
                + "name to an attribute object (see <a href=\"#modules.attr\">attr</a> "
                + "module). Attributes starting with <code>_</code> are private, and can be "
                + "used to add an implicit dependency on a label to a file (a repository "
                + "rule cannot depend on a generated artifact). The attribute "
                + "<code>name</code> is implicitly added and must not be specified."
      ),
      @Param(
        name = "local",
        type = Boolean.class,
        defaultValue = "False",
        doc =
            "Indicate that this rule fetches everything from the local system and should be "
                + "reevaluated at every fetch."
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction repositoryRule =
      new BuiltinFunction("repository_rule") {
        @SuppressWarnings({"rawtypes", "unused"})
        // an Attribute.Builder instead of a Attribute.Builder<?> but it's OK.
        public BaseFunction invoke(
            BaseFunction implementation,
            Object attrs,
            Boolean local,
            FuncallExpression ast,
            com.google.devtools.build.lib.syntax.Environment funcallEnv)
            throws EvalException {
          funcallEnv.checkLoadingPhase("repository_rule", ast.getLocation());
          // We'll set the name later, pass the empty string for now.
          Builder builder = new Builder("", RuleClassType.WORKSPACE, true);

          if (attrs != Runtime.NONE) {
            for (Map.Entry<String, Descriptor> attr :
                castMap(attrs, String.class, Descriptor.class, "attrs").entrySet()) {
              Descriptor attrDescriptor = attr.getValue();
              String attrName =
                  SkylarkRuleClassFunctions.attributeToNative(
                      attr.getKey(),
                      ast.getLocation(),
                      attrDescriptor.getAttributeBuilder().hasLateBoundValue());
              Attribute.Builder<?> attrBuilder = attrDescriptor.getAttributeBuilder();
              builder.addOrOverrideAttribute(attrBuilder.build(attrName));
            }
          }
          builder.addOrOverrideAttribute(attr("$local", BOOLEAN).defaultValue(local).build());
          builder.setConfiguredTargetFunction(implementation);
          builder.setRuleDefinitionEnvironment(funcallEnv);
          builder.setWorkspaceOnly();
          return new RepositoryRuleFunction(builder);
        }
      };

  private static final class RepositoryRuleFunction extends BaseFunction {
    private final Builder builder;

    public RepositoryRuleFunction(Builder builder) {
      super("repository_rule", FunctionSignature.KWARGS);
      this.builder = builder;
    }

    @Override
    public Object call(
        Object[] args, FuncallExpression ast, com.google.devtools.build.lib.syntax.Environment env)
        throws EvalException, InterruptedException {
      String ruleClassName = ast.getFunction().getName();
      try {
        if (ruleClassName.startsWith("_")) {
          throw new EvalException(
              ast.getLocation(),
              "Invalid rule class name '" + ruleClassName + "', cannot be private");
        }
        RuleClass ruleClass = builder.build(ruleClassName);
        PackageContext context = PackageFactory.getContext(env, ast);
        @SuppressWarnings("unchecked")
        Map<String, Object> attributeValues = (Map<String, Object>) args[0];
        return context
            .getBuilder()
            .externalPackageData()
            .createAndAddRepositoryRule(
                context.getBuilder(), ruleClass, null, attributeValues, ast);
      } catch (InvalidRuleException | NameConflictException | LabelSyntaxException e) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
    }
  }

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkRepositoryModule.class);
  }
}
