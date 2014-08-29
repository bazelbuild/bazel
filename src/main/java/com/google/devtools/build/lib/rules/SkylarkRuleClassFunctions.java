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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.syntax.SkylarkFunction.cast;
import static com.google.devtools.build.lib.syntax.SkylarkFunction.castList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.SkylarkLateBound;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunctionWithCallback;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunctionWithMap;
import com.google.devtools.build.lib.packages.MethodLibrary;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleFactory;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.packages.SkylarkFileType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
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
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Map;

/**
 * A helper class to provide an easier API for Skylark rule definitions.
 * This is experimental code.
 */
public class SkylarkRuleClassFunctions {

  @SkylarkBuiltin(name = "ANY_FILE", doc = "A file filter allowing any kind of file.")
  private static final FileTypeSet ANY_FILE = FileTypeSet.ANY_FILE;

  @SkylarkBuiltin(name = "NO_FILE", doc = "A file filter allowing no file at all.")
  private static final FileTypeSet NO_FILE = FileTypeSet.NO_FILE;

  @SkylarkBuiltin(name = "ANY_RULE",
      doc = "A rule class filter allowing any kind of rule class.")
  private static final Predicate<RuleClass> ANY_RULE = Attribute.ANY_RULE;

  @SkylarkBuiltin(name = "NO_RULE",
      doc = "A rule class filter allowing no rule class at all.")
  private static final Predicate<RuleClass> NO_RULE = Attribute.NO_RULE;

  //TODO(bazel-team): proper enum support
  @SkylarkBuiltin(name = "DATA_CFG", doc = "The default runfiles collection state.")
  private static final Object dataTransition = ConfigurationTransition.DATA;

  @SkylarkBuiltin(name = "HOST_CFG", doc = "The default runfiles collection state.")
  private static final Object hostTransition = ConfigurationTransition.HOST;

  @VisibleForTesting
  public static final Map<String, Object> JAVA_OBJECTS_TO_EXPOSE =
      ImmutableMap.<String, Object>builder()
          .put("ANY_FILE", ANY_FILE)
          .put("NO_FILE", NO_FILE)
          .put("ANY_RULE", ANY_RULE)
          .put("NO_RULE", NO_RULE)
          .put("DATA_CFG", dataTransition)
          .put("HOST_CFG", hostTransition)
          .put("Attr", SkylarkAttr.module)
          .build();

  private final SkylarkEnvironment env;

  private final RuleClass baseRule;
  private final ImmutableList<Function> builtInFunctions;
  private final PackageContext pkgContext;

  // TODO(bazel-team): Copied from ConfiguredRuleClassProvider for the transition from built-in
  // rules to skylark extensions. Using the same instance would require a large refactoring.
  // If we don't want to support old built-in rules and Skylark simultaneously
  // (except for transition phase) it's probably OK.
  private static LoadingCache<String, Label> labelCache = CacheBuilder.newBuilder().build(
      new CacheLoader<String, Label>() {
    @Override
    public Label load(String from) {
      try {
        return Label.parseAbsolute(from);
      } catch (Label.SyntaxException e) {
        throw new IllegalArgumentException(from);
      }
    }
  });

  private static final Attribute.ComputedDefault deprecationDefault =
      new Attribute.ComputedDefault() {
        @Override
        public Object getDefault(AttributeMap rule) {
          return rule.getPackageDefaultDeprecation();
        }
      };

  private static final Attribute.ComputedDefault testonlyDefault =
      new Attribute.ComputedDefault() {
        @Override
        public Object getDefault(AttributeMap rule) {
          return rule.getPackageDefaultTestOnly();
        }
     };

  private SkylarkRuleClassFunctions(SkylarkEnvironment env, PackageContext pkgContext) {
    this.env = Preconditions.checkNotNull(env);
    this.pkgContext = pkgContext;
    // TODO(bazel-team): we might want to define base rule in Skylark later.
    // Right now we need some default attributes.
    baseRule = new RuleClass.Builder("$base_rule", RuleClassType.ABSTRACT, true)
        .add(attr("deprecation", STRING).nonconfigurable().value(deprecationDefault))
        .add(attr("expect_failure", STRING))
        .add(attr("tags", STRING_LIST).orderIndependent().nonconfigurable().taggable())
        .add(attr("testonly", BOOLEAN).nonconfigurable().value(testonlyDefault))
        .add(attr("visibility", NODEP_LABEL_LIST).orderIndependent().nonconfigurable().cfg(HOST))
        .build();

    ImmutableList.Builder<Function> builtInFunctionsBuilder = ImmutableList.builder();
    SkylarkFunction.collectSkylarkFunctionsFromFields(getClass(), this, builtInFunctionsBuilder);
    builtInFunctions = builtInFunctionsBuilder.build();
  }

  public static SkylarkEnvironment getNewEnvironment(PackageContext pkgContext) {
    SkylarkEnvironment env = new SkylarkEnvironment();
    SkylarkRuleClassFunctions functions = new SkylarkRuleClassFunctions(env, pkgContext);
    for (Function builtInFunction : functions.builtInFunctions) {
      env.update(builtInFunction.getName(), builtInFunction);
    }

    for (Map.Entry<String, Object> entry : JAVA_OBJECTS_TO_EXPOSE.entrySet()) {
      env.update(entry.getKey(), entry.getValue());
    }
    SkylarkAttr.registerFunctions(env);

    MethodLibrary.setupMethodEnvironment(env);
    return env;
  }

  static Attribute.Builder<?> createAttribute(String strType, Map<String, Object> arguments,
      FuncallExpression ast, Environment funcallEnv)
      throws EvalException, ConversionException {
    final Location loc = ast.getLocation();
    Type<?> type = createTypeFromString(strType, loc, "invalid attribute type %s");
    // We use an empty name now so that we can set it later.
    // This trick makes sense only in the context of Skylark (builtin rules should not use it).
    Attribute.Builder<?> builder = Attribute.attr("", type);

    Object defaultValue = arguments.get("default");
    if (defaultValue != null) {
      if (defaultValue instanceof UserDefinedFunction) {
        // Late bound attribute
        UserDefinedFunction func =
            cast(defaultValue, UserDefinedFunction.class, "default", loc);
        final SkylarkCallbackFunction callback =
            new SkylarkCallbackFunction(func, ast, (SkylarkEnvironment) funcallEnv);
        final SkylarkLateBound computedValue;
        if (type.equals(Type.LABEL) || type.equals(Type.LABEL_LIST)) {
          computedValue = new SkylarkLateBound(false, callback);
        } else {
          throw new EvalException(loc, "Only label type attributes can be late bound");
        }
        builder.value(computedValue);
      } else {
        builder.defaultValue(defaultValue);
      }
    }

    for (String flag :
             castList(arguments.get("flags"), String.class, "flags for attribute definition")) {
      builder.setPropertyFlag(flag);
    }

    if (arguments.containsKey("file_types")) {
      Object fileTypesObj = arguments.get("file_types");
      if (fileTypesObj == FileTypeSet.ANY_FILE || fileTypesObj == FileTypeSet.NO_FILE) {
        builder.allowedFileTypes((FileTypeSet) fileTypesObj);
      } else if (fileTypesObj instanceof SkylarkFileType) {
        builder.allowedFileTypes(((SkylarkFileType) fileTypesObj).getFileTypeSet());
      } else {
        builder.allowedFileTypes(FileTypeSet.of(Iterables.transform(
            castList(fileTypesObj, String.class, "allowed file types for attribute definition"),
            new com.google.common.base.Function<String, FileType>() {
              @Override
                public FileType apply(String input) {
                return FileType.of(input);
              }
            })));
      }
    }

    Object ruleClassesObj = arguments.get("rule_classes");
    if (ruleClassesObj == Attribute.ANY_RULE || ruleClassesObj == Attribute.NO_RULE) {
      // This causes an unchecked warning but it's fine because of the surrounding if.
      builder.allowedRuleClasses((Predicate<RuleClass>) ruleClassesObj);
    } else if (ruleClassesObj != null) {
      builder.allowedRuleClasses(castList(ruleClassesObj, String.class,
              "allowed rule classes for attribute definition"));
    }

    if (arguments.containsKey("cfg")) {
      builder.cfg(
          cast(arguments.get("cfg"), ConfigurationTransition.class, "configuration", loc));
    }
    return builder;
  }

  // TODO(bazel-team): implement attribute copy and other rule properties

  @SkylarkBuiltin(name = "rule", doc = "Creates a rule class.",
      returnType = Rule.class,
      mandatoryParams = {
      @Param(name = "implementation", type = UserDefinedFunction.class,
          doc = "the function implementing this rule, has to have exactly one parameter: 'ctx'")},
      optionalParams = {
      @Param(name = "type", type = String.class, doc = ""),
      @Param(name = "parents", type = List.class,
          doc = "list of parent rule classes, this rule class inherits all the attributes and "
              + "the impicit outputs of the parent rule classes"),
      @Param(name = "attr", doc = "dictionary mapping an attribute name to an attribute"),
      @Param(name = "implicit_outputs", doc = "implicit outputs of this rule")})
  private final SkylarkFunction rule = new SkylarkFunction("rule") {

        @Override
        public Object call(Map<String, Object> arguments, FuncallExpression ast,
            Environment funcallEnv) throws EvalException, ConversionException {
          final Location loc = ast.getLocation();

          RuleClassType type = RuleClassType.NORMAL;
          if (arguments.containsKey("type")) {
            type =
                RuleClassType.valueOf(cast(arguments.get("type"), String.class, "rule type", loc));
          }

          // We'll set the name later, pass the empty string for now.
          final RuleClass.Builder builder = new RuleClass.Builder("", type, true, baseRule);

          for (Map.Entry<String, Attribute.Builder> attr :
                   castMap(arguments.get("attr"), String.class, Attribute.Builder.class, "attr")) {
            String attrName = attr.getKey();
            Attribute.Builder<?> attrBuilder = attr.getValue();
            attrBuilder.setName(attrName);
            builder.addOrOverrideAttribute(attrBuilder.build());
          }

          if (arguments.containsKey("implicit_outputs")) {
            final Object implicitOutputs = arguments.get("implicit_outputs");
            if (implicitOutputs instanceof UserDefinedFunction) {
              UserDefinedFunction func = (UserDefinedFunction) implicitOutputs;
              final SkylarkCallbackFunction callback =
                  new SkylarkCallbackFunction(func, ast, (SkylarkEnvironment) funcallEnv);
              builder.setImplicitOutputsFunction(
                  new SkylarkImplicitOutputsFunctionWithCallback(callback, loc));
            } else {
              builder.setImplicitOutputsFunction(new SkylarkImplicitOutputsFunctionWithMap(
                  toMap(castMap(arguments.get("implicit_outputs"), String.class, String.class,
                  "implicit outputs of the rule class"))));
            }
          }

          builder.setConfiguredTargetFunction(cast(arguments.get("implementation"),
              UserDefinedFunction.class, "rule implementation", loc));
          builder.setRuleDefinitionEnvironment(env);
          return new RuleFunction(builder);
        }
      };

  // This class is needed for testing
  final class RuleFunction extends AbstractFunction {
    // Note that this means that we can reuse the same builder.
    // This is fine since we change only the name.
    private final RuleClass.Builder builder;

    public RuleFunction(Builder builder) {
      super("rule");
      this.builder = builder;
    }

    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      try {
        builder.setName(ast.getFunction().getName());
        RuleClass ruleClass = builder.build();
        return RuleFactory.createAndAddRule(pkgContext, ruleClass, kwargs, ast);
      } catch (InvalidRuleException | NameConflictException e) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
    }

    @VisibleForTesting
    RuleClass.Builder getBuilder() {
      return builder;
    }
  }

  @SkylarkBuiltin(name = "label", doc = "Creates a label referring to a BUILD target.",
      returnType = Label.class,
      mandatoryParams = {@Param(name = "label", type = String.class, doc = "the label string")})
  private final SkylarkFunction label = new SimpleSkylarkFunction("label") {
        @Override
        public Object call(Map<String, Object> arguments, Location loc) throws EvalException,
            ConversionException {
          String label = cast(arguments.get("label"), String.class, "label", loc);
          return labelCache.getUnchecked(label);
        }
      };

  @SkylarkBuiltin(name = "filetype", doc = "Creates a file filter from a list of strings.",
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

  private static Type<?> createTypeFromString(
      String typeString, Location location, String errorMsg) throws EvalException {
    try {
      Field field = Type.class.getField(typeString);
      if (Type.class.isAssignableFrom(field.getType()) && Modifier.isPublic(field.getModifiers())) {
        return (Type<?>) field.get(null);
      } else {
        throw new EvalException(location, String.format(errorMsg, typeString));
      }
    } catch (IllegalArgumentException | SecurityException | IllegalAccessException e) {
      throw new EvalException(location, e.getMessage());
    } catch (NoSuchFieldException e) {
      throw new EvalException(location, String.format(errorMsg, typeString));
    }
  }

  @VisibleForTesting
  public static Iterable<Function> getStaticBuiltInFunctions() {
    return ImmutableList.of((Function) fileType);
  }
}
