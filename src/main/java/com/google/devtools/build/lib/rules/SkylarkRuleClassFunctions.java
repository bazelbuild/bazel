// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.DATA;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunctionWithCallback;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunctionWithMap;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleFactory;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.packages.SkylarkFileType;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.NoSuchVariableException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkCallbackFunction;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkFunction.SimpleSkylarkFunction;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.RunUnder;

import java.util.List;
import java.util.Map;

/**
 * A helper class to provide an easier API for Skylark rule definitions.
 * This is experimental code.
 */
public class SkylarkRuleClassFunctions {

  //TODO(bazel-team): proper enum support
  @SkylarkBuiltin(name = "DATA_CFG", doc = "The default runfiles collection state.")
  private static final Object dataTransition = ConfigurationTransition.DATA;

  @SkylarkBuiltin(name = "HOST_CFG", doc = "The default runfiles collection state.")
  private static final Object hostTransition = ConfigurationTransition.HOST;

  private static final Attribute.ComputedDefault DEPRECATION =
      new Attribute.ComputedDefault() {
        @Override
        public Object getDefault(AttributeMap rule) {
          return rule.getPackageDefaultDeprecation();
        }
      };

  private static final Attribute.ComputedDefault TEST_ONLY =
      new Attribute.ComputedDefault() {
        @Override
        public Object getDefault(AttributeMap rule) {
          return rule.getPackageDefaultTestOnly();
        }
     };

  private static final LateBoundLabel<BuildConfiguration> RUN_UNDER =
      new LateBoundLabel<BuildConfiguration>() {
        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          RunUnder runUnder = configuration.getRunUnder();
          return runUnder == null ? null : runUnder.getLabel();
        }
      };

  // TODO(bazel-team): Copied from ConfiguredRuleClassProvider for the transition from built-in
  // rules to skylark extensions. Using the same instance would require a large refactoring.
  // If we don't want to support old built-in rules and Skylark simultaneously
  // (except for transition phase) it's probably OK.
  private static LoadingCache<String, Label> labelCache =
      CacheBuilder.newBuilder().build(new CacheLoader<String, Label>() {
    @Override
    public Label load(String from) {
      try {
        return Label.parseAbsolute(from);
      } catch (Label.SyntaxException e) {
        throw new IllegalArgumentException(from);
      }
    }
  });

  // TODO(bazel-team): Remove the code duplication (BaseRuleClasses and this class).
  private static final RuleClass baseRule =
      new RuleClass.Builder("$base_rule", RuleClassType.ABSTRACT, true)
          .add(attr("deprecation", STRING).nonconfigurable().value(DEPRECATION))
          .add(attr("expect_failure", STRING))
          .add(attr("generator_name", STRING).undocumented("internal"))
          .add(attr("generator_function", STRING).undocumented("internal"))
          .add(attr("tags", STRING_LIST).orderIndependent().nonconfigurable().taggable())
          .add(attr("testonly", BOOLEAN).nonconfigurable().value(TEST_ONLY))
          .add(attr("visibility", NODEP_LABEL_LIST).orderIndependent().nonconfigurable().cfg(HOST))
          .build();

  private static final RuleClass testBaseRule =
      new RuleClass.Builder("$test_base_rule", RuleClassType.ABSTRACT, true, baseRule)
          .add(attr("size", STRING).value("medium").taggable().nonconfigurable())
          .add(attr("timeout", STRING).taggable().nonconfigurable().value(
              new Attribute.ComputedDefault() {
                @Override
                public Object getDefault(AttributeMap rule) {
                  TestSize size = TestSize.getTestSize(rule.get("size", Type.STRING));
                  if (size != null) {
                    String timeout = size.getDefaultTimeout().toString();
                    if (timeout != null) {
                      return timeout;
                    }
                  }
                  return "illegal";
                }
              }))
          .add(attr("flaky", BOOLEAN).value(false).taggable().nonconfigurable())
          .add(attr("shard_count", INTEGER).value(-1))
          .add(attr("env", STRING_LIST).value(ImmutableList.of("corp"))
               .undocumented("Deprecated").taggable().nonconfigurable())
          .add(attr("local", BOOLEAN).value(false).taggable().nonconfigurable())
          .add(attr("$test_runtime", LABEL_LIST).cfg(HOST).value(ImmutableList.of(
              labelCache.getUnchecked("//tools/test:runtime"))))
          .add(attr(":run_under", LABEL).cfg(DATA).value(RUN_UNDER))
          .build();

  /**
   * In native code, private values start with $.
   * In Skylark, private values start with _, because of the grammar.
   */
  private static String attributeToNative(String oldName) {
    if (!oldName.isEmpty() && oldName.charAt(0) == '_') {
      return "$" + oldName.substring(1);
    }
    return oldName;
  }

  // TODO(bazel-team): implement attribute copy and other rule properties

  @SkylarkBuiltin(name = "rule", doc =
      "Creates a new rule. Store it in a global value, so that it can be loaded and called "
      + "from BUILD files.",
      onlyLoadingPhase = true,
      returnType = Function.class,
      mandatoryParams = {
      @Param(name = "implementation", type = UserDefinedFunction.class,
          doc = "the function implementing this rule, has to have exactly one parameter: "
             + "<code>ctx</code>. The function is called during analysis phase for each "
             + "instance of the rule. It can access the attributes provided by the user. "
             + "It must create actions to generate all the declared outputs.")
      },
      optionalParams = {
      @Param(name = "test", type = Boolean.class, doc = "Whether this rule is a test rule. "
             + "If True, the rule must end with <code>_test</code> (otherwise it cannot)."),
      @Param(name = "attr", doc =
          "dictionary to declare all the attributes of the rule. It maps from an attribute name "
          + "to an attribute object (see 'attr' module). Attributes starting with <code>_</code> "
          + "are private, and can be used to add an implicit dependency on a label."),
      @Param(name = "outputs", doc = "outputs of this rule. "
          + "It is a dictionary mapping from string to a template name. For example: "
          + "<code>{\"ext\": \"${name}.ext\"}</code>. <br>"
          // TODO(bazel-team): Make doc more clear, wrt late-bound attributes.
          + "It may also be a function (which receives <code>ctx.attr</code> as argument) "
          + "returning such a dictionary."),
      @Param(name = "executable", type = Boolean.class,
          doc = "whether this rule always outputs an executable of the same name or not. If True, "
          + "there must be an action that generates <code>ctx.outputs.executable</code>.")})
  private static final SkylarkFunction rule = new SkylarkFunction("rule") {

        @Override
        public Object call(Map<String, Object> arguments, FuncallExpression ast,
            Environment funcallEnv) throws EvalException, ConversionException {
          final Location loc = ast.getLocation();

          RuleClassType type = RuleClassType.NORMAL;
          if (arguments.containsKey("test") && EvalUtils.toBoolean(arguments.get("test"))) {
            type = RuleClassType.TEST;
          }

          // We'll set the name later, pass the empty string for now.
          final RuleClass.Builder builder = type == RuleClassType.TEST
              ? new RuleClass.Builder("", type, true, testBaseRule)
              : new RuleClass.Builder("", type, true, baseRule);

          for (Map.Entry<String, Attribute.Builder> attr :
                   castMap(arguments.get("attr"), String.class, Attribute.Builder.class, "attr")) {
            String attrName = attributeToNative(attr.getKey());
            Attribute.Builder<?> attrBuilder = attr.getValue();
            builder.addOrOverrideAttribute(attrBuilder.build(attrName));
          }
          if (arguments.containsKey("executable") && (Boolean) arguments.get("executable")) {
            builder.addOrOverrideAttribute(
                attr("$is_executable", BOOLEAN).nonconfigurable().value(true).build());
            builder.setOutputsDefaultExecutable();
          }

          if (arguments.containsKey("outputs")) {
            final Object implicitOutputs = arguments.get("outputs");
            if (implicitOutputs instanceof UserDefinedFunction) {
              UserDefinedFunction func = (UserDefinedFunction) implicitOutputs;
              final SkylarkCallbackFunction callback =
                  new SkylarkCallbackFunction(func, ast, (SkylarkEnvironment) funcallEnv);
              builder.setImplicitOutputsFunction(
                  new SkylarkImplicitOutputsFunctionWithCallback(callback, loc));
            } else {
              builder.setImplicitOutputsFunction(new SkylarkImplicitOutputsFunctionWithMap(
                  toMap(castMap(arguments.get("outputs"), String.class, String.class,
                  "implicit outputs of the rule class"))));
            }
          }

          builder.setConfiguredTargetFunction(
              (UserDefinedFunction) arguments.get("implementation"));
          builder.setRuleDefinitionEnvironment((SkylarkEnvironment) funcallEnv);
          return new RuleFunction(builder, type);
        }
      };

  // This class is needed for testing
  static final class RuleFunction extends AbstractFunction {
    // Note that this means that we can reuse the same builder.
    // This is fine since we don't modify the builder from here.
    private final RuleClass.Builder builder;
    private final RuleClassType type;

    public RuleFunction(Builder builder, RuleClassType type) {
      super("rule");
      this.builder = builder;
      this.type = type;
    }

    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      try {
        String ruleClassName = ast.getFunction().getName();
        if (type == RuleClassType.TEST != TargetUtils.isTestRuleName(ruleClassName)) {
          throw new EvalException(ast.getLocation(), "Invalid rule class name '" + ruleClassName
              + "', test rule class names must end with '_test' and other rule classes must not");
        }
        RuleClass ruleClass = builder.build(ruleClassName);
        PackageContext pkgContext = (PackageContext) env.lookup(PackageFactory.PKG_CONTEXT);
        return RuleFactory.createAndAddRule(pkgContext, ruleClass, kwargs, ast);
      } catch (InvalidRuleException | NameConflictException | NoSuchVariableException e) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
    }

    @VisibleForTesting
    RuleClass.Builder getBuilder() {
      return builder;
    }
  }

  @SkylarkBuiltin(name = "label", doc = "Creates a label referring to a BUILD target. Use "
      + "this function only when you want to give a default value for the label attributes. "
      + "Example: <br><pre class=code>label(\"//tools:default\")</pre>",
      returnType = Label.class,
      mandatoryParams = {@Param(name = "label", type = String.class, doc = "the label string")})
  private static final SkylarkFunction label = new SimpleSkylarkFunction("label") {
        @Override
        public Object call(Map<String, Object> arguments, Location loc) throws EvalException,
            ConversionException {
          return labelCache.getUnchecked((String) arguments.get("label"));
        }
      };

  @SkylarkBuiltin(name = "filetype",
      doc = "Creates a file filter from a list of strings. For example, to match files ending "
      + "with .cc or .cpp, use: <pre class=code>filetype([\".cc\", \".cpp\"])</pre>",
      returnType = SkylarkFileType.class,
      mandatoryParams = {
      @Param(name = "types", doc = "a list of the accepted file extensions")})
  private static final SkylarkFunction fileType = new SimpleSkylarkFunction("filetype") {
        @Override
        public Object call(Map<String, Object> arguments, Location loc) throws EvalException,
            ConversionException {
          return SkylarkFileType.of(castList(arguments.get("types"), String.class, "file types"));
        }
      };
}
