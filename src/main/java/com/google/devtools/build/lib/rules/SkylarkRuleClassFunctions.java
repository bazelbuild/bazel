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
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.syntax.SkylarkType.castList;
import static com.google.devtools.build.lib.syntax.SkylarkType.castMap;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.RunUnder;
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
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
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
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/**
 * A helper class to provide an easier API for Skylark rule definitions.
 * This is experimental code.
 */
public class SkylarkRuleClassFunctions {

  //TODO(bazel-team): proper enum support
  @SkylarkBuiltin(name = "DATA_CFG", returnType = ConfigurationTransition.class,
      doc = "Experimental. Specifies a transition to the data configuration.")
  private static final Object dataTransition = ConfigurationTransition.DATA;

  @SkylarkBuiltin(name = "HOST_CFG", returnType = ConfigurationTransition.class,
      doc = "Specifies a transition to the host configuration.")
  private static final Object hostTransition = ConfigurationTransition.HOST;

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
    public Label load(String from) throws Exception {
      try {
        return Label.parseAbsolute(from);
      } catch (Label.SyntaxException e) {
        throw new Exception(from);
      }
    }
  });

  // TODO(bazel-team): Remove the code duplication (BaseRuleClasses and this class).
  private static final RuleClass baseRule =
      BaseRuleClasses.commonCoreAndSkylarkAttributes(
          new RuleClass.Builder("$base_rule", RuleClassType.ABSTRACT, true))
          .add(attr("expect_failure", STRING))
          .build();

  private static final RuleClass testBaseRule =
      new RuleClass.Builder("$test_base_rule", RuleClassType.ABSTRACT, true, baseRule)
          .add(attr("size", STRING).value("medium").taggable()
              .nonconfigurable("used in loading phase rule validation logic"))
          .add(attr("timeout", STRING).taggable()
              .nonconfigurable("used in loading phase rule validation logic").value(
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
          .add(attr("flaky", BOOLEAN).value(false).taggable()
              .nonconfigurable("taggable - called in Rule.getRuleTags"))
          .add(attr("shard_count", INTEGER).value(-1))
          .add(attr("local", BOOLEAN).value(false).taggable()
              .nonconfigurable("policy decision: this should be consistent across configurations"))
          .add(attr("$test_runtime", LABEL_LIST).cfg(HOST).value(ImmutableList.of(
              labelCache.getUnchecked("//tools/test:runtime"))))
          .add(attr(":run_under", LABEL).cfg(DATA).value(RUN_UNDER))
          .build();

  /**
   * In native code, private values start with $.
   * In Skylark, private values start with _, because of the grammar.
   */
  private static String attributeToNative(String oldName, Location loc, boolean isLateBound)
      throws EvalException {
    if (oldName.isEmpty()) {
      throw new EvalException(loc, "Attribute name cannot be empty");
    }
    if (isLateBound) {
      if (oldName.charAt(0) != '_') {
        throw new EvalException(loc, "When an attribute value is a function, "
            + "the attribute must be private (start with '_')");
      }
      return ":" + oldName.substring(1);
    }
    if (oldName.charAt(0) == '_') {
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
      @Param(name = "attrs", type = Map.class, doc =
          "dictionary to declare all the attributes of the rule. It maps from an attribute name "
          + "to an attribute object (see <a href=\"#modules.attr\">attr</a> module). "
          + "Attributes starting with <code>_</code> are private, and can be used to add "
          + "an implicit dependency on a label."),
      @Param(name = "outputs", doc = "outputs of this rule. "
          + "It is a dictionary mapping from string to a template name. "
          + "For example: <code>{\"ext\": \"${name}.ext\"}</code>. <br>"
          + "The dictionary key becomes a field in <code>ctx.outputs</code>. "
          // TODO(bazel-team): Make doc more clear, wrt late-bound attributes.
          + "It may also be a function (which receives <code>ctx.attr</code> as argument) "
          + "returning such a dictionary."),
      @Param(name = "executable", type = Boolean.class,
          doc = "whether this rule always outputs an executable of the same name or not. If True, "
          + "there must be an action that generates <code>ctx.outputs.executable</code>."),
      @Param(name = "output_to_genfiles", type = Boolean.class,
          doc = "If true, the files will be generated in the genfiles directory instead of the "
          + "bin directory. This is used for compatibility with existing rules.")})
  private static final SkylarkFunction rule = new SkylarkFunction("rule") {

        @Override
        @SuppressWarnings({"rawtypes", "unchecked"}) // castMap produces
        // an Attribute.Builder instead of a Attribute.Builder<?> but it's OK.
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

          for (Map.Entry<String, Attribute.Builder> attr : castMap(arguments.get("attrs"),
              String.class, Attribute.Builder.class, "attrs").entrySet()) {
            Attribute.Builder<?> attrBuilder = attr.getValue();
            String attrName = attributeToNative(attr.getKey(), loc,
                attrBuilder.hasLateBoundValue());
            builder.addOrOverrideAttribute(attrBuilder.build(attrName));
          }
          if (arguments.containsKey("executable") && (Boolean) arguments.get("executable")) {
            builder.addOrOverrideAttribute(
                attr("$is_executable", BOOLEAN).value(true)
                    .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target")
                    .build());
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
                  ImmutableMap.copyOf(castMap(arguments.get("outputs"), String.class, String.class,
                      "implicit outputs of the rule class"))));
            }
          }

          if (arguments.containsKey("output_to_genfiles")
              && (Boolean) arguments.get("output_to_genfiles")) {
            builder.setOutputToGenfiles();
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

    @SuppressWarnings("unchecked") // the magic hidden $pkg_context variable is guaranteed
    // to be a PackageContext
    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      try {
        String ruleClassName = ast.getFunction().getName();
        if (ruleClassName.startsWith("_")) {
          throw new EvalException(ast.getLocation(), "Invalid rule class name '" + ruleClassName
              + "', cannot be private");
        }
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

  @SkylarkBuiltin(name = "Label", doc = "Creates a Label referring to a BUILD target. Use "
      + "this function only when you want to give a default value for the label attributes. "
      + "Example: <br><pre class=language-python>Label(\"//tools:default\")</pre>",
      returnType = Label.class,
      mandatoryParams = {@Param(name = "label_string", type = String.class,
            doc = "the label string")})
  private static final SkylarkFunction label = new SimpleSkylarkFunction("Label") {
        @Override
        public Object call(Map<String, Object> arguments, Location loc) throws EvalException,
            ConversionException {
          String labelString = (String) arguments.get("label_string");
          try {
            return labelCache.get(labelString);
          } catch (ExecutionException e) {
            throw new EvalException(loc, "Illegal absolute label syntax: " + labelString);
          }
        }
      };

  @SkylarkBuiltin(name = "FileType",
      doc = "Creates a file filter from a list of strings. For example, to match files ending "
      + "with .cc or .cpp, use: <pre class=language-python>FileType([\".cc\", \".cpp\"])</pre>",
      returnType = SkylarkFileType.class,
      mandatoryParams = {
      @Param(name = "types", type = SkylarkList.class, generic1 = String.class,
          doc = "a list of the accepted file extensions")})
  private static final SkylarkFunction fileType = new SimpleSkylarkFunction("FileType") {
        @Override
        public Object call(Map<String, Object> arguments, Location loc) throws EvalException,
            ConversionException {
          return SkylarkFileType.of(castList(arguments.get("types"), String.class));
        }
      };

  @SkylarkBuiltin(name = "to_proto",
      doc = "Creates a text message from the struct parameter. This method only works if all "
          + "struct elements (recursively) are strings, ints, booleans, other structs or a "
          + "list of these types. Quotes and new lines in strings are escaped. "
          + "Examples:<br><pre class=language-python>"
          + "struct(key=123).to_proto()\n# key: 123\n\n"
          + "struct(key=True).to_proto()\n# key: true\n\n"
          + "struct(key=[1, 2, 3]).to_proto()\n# key: 1\n# key: 2\n# key: 3\n\n"
          + "struct(key='text').to_proto()\n# key: \"text\"\n\n"
          + "struct(key=struct(inner_key='text')).to_proto()\n"
          + "# key {\n#   inner_key: \"text\"\n# }\n\n"
          + "struct(key=[struct(inner_key=1), struct(inner_key=2)]).to_proto()\n"
          + "# key {\n#   inner_key: 1\n# }\n# key {\n#   inner_key: 2\n# }\n\n"
          + "struct(key=struct(inner_key=struct(inner_inner_key='text'))).to_proto()\n"
          + "# key {\n#    inner_key {\n#     inner_inner_key: \"text\"\n#   }\n# }\n</pre>",
      objectType = SkylarkClassObject.class, returnType = String.class)
  private static final SkylarkFunction toProto = new SimpleSkylarkFunction("to_proto") {
    @Override
    public Object call(Map<String, Object> arguments, Location loc) throws EvalException,
        ConversionException {
      ClassObject self = (ClassObject) arguments.get("self");
      StringBuilder sb = new StringBuilder();
      printTextMessage(self, sb, 0, loc);
      return sb.toString();
    }

    private void printTextMessage(ClassObject object, StringBuilder sb,
        int indent, Location loc) throws EvalException {
      for (String key : object.getKeys()) {
        printTextMessage(key, object.getValue(key), sb, indent, loc);
      }
    }

    private void printSimpleTextMessage(String key, Object value, StringBuilder sb,
        int indent, Location loc, String container) throws EvalException {
      if (value instanceof ClassObject) {
        print(sb, key + " {", indent);
        printTextMessage((ClassObject) value, sb, indent + 1, loc);
        print(sb, "}", indent);
      } else if (value instanceof String) {
        print(sb, key + ": \"" + escape((String) value) + "\"", indent);
      } else if (value instanceof Integer) {
        print(sb, key + ": " + value, indent);
      } else if (value instanceof Boolean) {
        // We're relying on the fact that Java converts Booleans to Strings in the same way
        // as the protocol buffers do.
        print(sb, key + ": " + value, indent);
      } else {
        throw new EvalException(loc,
            "Invalid text format, expected a struct, a string, a bool, or an int but got a "
            + EvalUtils.getDataTypeName(value) + " for " + container + " '" + key + "'");
      }
    }

    private void printTextMessage(String key, Object value, StringBuilder sb,
        int indent, Location loc) throws EvalException {
      if (value instanceof SkylarkList) {
        for (Object item : ((SkylarkList) value)) {
          // TODO(bazel-team): There should be some constraint on the fields of the structs
          // in the same list but we ignore that for now.
          printSimpleTextMessage(key, item, sb, indent, loc, "list element in struct field");
        }
      } else {
        printSimpleTextMessage(key, value, sb, indent, loc, "struct field");
      }
    }

    private String escape(String string) {
      // TODO(bazel-team): use guava's SourceCodeEscapers when it's released.
      return string.replace("\"", "\\\"").replace("\n", "\\n");
    }

    private void print(StringBuilder sb, String text, int indent) {
      for (int i = 0; i < indent; i++) {
        sb.append("  ");
      }
      sb.append(text);
      sb.append("\n");
    }
  };
}
