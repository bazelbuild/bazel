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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.DATA;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

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
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.MethodLibrary;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.SkylarkFileType;
import com.google.devtools.build.lib.packages.SkylarkRuleFactory;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.Environment;
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
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.Path;

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

  @SkylarkBuiltin(name = "Attr", doc = "Module for creating new attributes.")
  private static final Object ATTR = SkylarkAttr.module;

  @VisibleForTesting
  static final Map<String, Object> JAVA_OBJECTS_TO_EXPOSE =
      ImmutableMap.<String, Object>builder()
          .put("ANY_FILE", ANY_FILE)
          .put("NO_FILE", NO_FILE)
          .put("ANY_RULE", ANY_RULE)
          .put("NO_RULE", NO_RULE)
          .put("DATA_CFG", dataTransition)
          .put("HOST_CFG", hostTransition)
          .put("Attr", ATTR)
          .build();

  private final SkylarkRuleFactory ruleFactory;
  private final SkylarkEnvironment env;
  private final Path file;

  private final RuleClass baseRule;
  private final ImmutableList<Function> builtInFunctions;

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

  private SkylarkRuleClassFunctions(
      SkylarkRuleFactory ruleFactory, SkylarkEnvironment env, Path file) {
    this.ruleFactory = Preconditions.checkNotNull(ruleFactory);
    this.env = Preconditions.checkNotNull(env);
    this.file = Preconditions.checkNotNull(file);
    // TODO(bazel-team): we might want to define base rule in Skylark later.
    // Right now we need some default attributes.
    baseRule = new RuleClass.Builder("$base_rule", RuleClassType.ABSTRACT, true)
        .add(attr("deprecation", STRING).nonconfigurable().value(deprecationDefault))
        .add(attr("visibility", NODEP_LABEL_LIST).orderIndependent().nonconfigurable().cfg(HOST))
        .add(attr("tags", STRING_LIST).orderIndependent().nonconfigurable().taggable())
        .add(attr("deps", LABEL_LIST))
        .add(attr("data", LABEL_LIST).cfg(DATA))
        .add(attr("testonly", BOOLEAN).nonconfigurable().value(testonlyDefault))
        .build();

    ImmutableList.Builder<Function> builtInFunctionsBuilder = ImmutableList.builder();
    SkylarkFunction.collectSkylarkFunctionsFromFields(this, builtInFunctionsBuilder);
    builtInFunctions = builtInFunctionsBuilder.build();
  }

  public static SkylarkEnvironment getNewEnvironment(SkylarkRuleFactory ruleFactory, Path file) {
    SkylarkEnvironment env = new SkylarkEnvironment();
    SkylarkRuleClassFunctions functions = new SkylarkRuleClassFunctions(ruleFactory, env, file);
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

  // TODO(bazel-team): Get rid of this function, and use Attr.* functions.
  @SkylarkBuiltin(name = "attr", doc = "Creates a rule class attribute.",
      mandatoryParams = {
      @Param(name = "type", type = String.class, doc = "type of the attribute")},
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "file_types", type = FileTypeSet.class,
          doc = "allowed file types of the label type attribute"),
      @Param(name = "rule_classes", doc = "allowed rule classes of the label type attribute"),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  private static final SkylarkFunction attr = new SkylarkFunction("attr") {
    @SuppressWarnings("unchecked")
    @Override
    public Object call(Map<String, Object> arguments, FuncallExpression ast,
        Environment funcallEnv) throws EvalException, ConversionException {
      String type = cast(arguments.get("type"), String.class, "attribute type", ast.getLocation());
      return createAttribute(type, arguments, ast, funcallEnv);
    }
  };

  // TODO(bazel-team): implement attribute copy and other rule properties

  @SkylarkBuiltin(name = "rule", doc = "Creates a rule class.",
      mandatoryParams = {
      @Param(name = "name", type = String.class, doc = "name of the rule class"),
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
          String name = cast(arguments.get("name"), String.class, "rule class name", loc);

          RuleClassType type = RuleClassType.NORMAL;
          if (arguments.containsKey("type")) {
            type =
                RuleClassType.valueOf(cast(arguments.get("type"), String.class, "rule type", loc));
          }

          RuleClass[] parents = Iterables.toArray(Iterables.concat(
              castList(arguments.get("parents"), RuleClass.class,
                  "parent rule classes of the rule class"),
              ImmutableList.of(baseRule)), RuleClass.class);

          RuleClass.Builder builder = new RuleClass.Builder(name, type, true, parents);

          for (Map.Entry<String, Attribute.Builder> attr :
                   castMap(arguments.get("attr"), String.class, Attribute.Builder.class, "attr")) {
            String attrName = attr.getKey();
            if (attr.getValue() == null) {
              builder.removeAttribute(attrName);
            } else {
              Attribute.Builder<?> attrBuilder = attr.getValue();
              attrBuilder.setName(attrName);
              builder.addOrOverrideAttribute(attrBuilder.build());
            }
          }

          if (arguments.containsKey("implicit_outputs")) {
            final Object implicitOutputs = arguments.get("implicit_outputs");
            if (implicitOutputs instanceof UserDefinedFunction) {
              UserDefinedFunction func = (UserDefinedFunction) implicitOutputs;
              final SkylarkCallbackFunction callback =
                  new SkylarkCallbackFunction(func, ast, (SkylarkEnvironment) funcallEnv);
              builder.setImplicitOutputsFunction(new SkylarkImplicitOutputsFunction(callback, loc));
            } else {
              builder.setImplicitOutputsFunction(ImplicitOutputsFunction.fromTemplates(castList(
                  arguments.get("implicit_outputs"), String.class,
                  "implicit outputs of the rule class")));
            }
          }

          builder.setConfiguredTargetFunction(cast(arguments.get("implementation"),
              UserDefinedFunction.class, "rule implementation", loc));
          builder.setRuleDefinitionEnvironment(env);

          RuleClass ruleClass = builder.build();
          ruleFactory.addSkylarkRuleClass(ruleClass, file);
          return ruleClass;
        }
      };

  @SkylarkBuiltin(name = "label", doc = "Creates a label referring to a BUILD target.",
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
      mandatoryParams = {
      @Param(name = "types", doc = "a list of the accepted file extensions")})
  private static final SkylarkFunction fileType = new SimpleSkylarkFunction("filetype") {
        @Override
        public Object call(Map<String, Object> arguments, Location loc) throws EvalException,
            ConversionException {
          return SkylarkFileType.of(castList(arguments.get("types"), String.class, "file types"));
        }
      };

  public static <TYPE> Iterable<TYPE> castList(
      Object obj, final Class<TYPE> type, final String what) throws ConversionException {
    if (obj == null) {
      return ImmutableList.of();
    }
    return Iterables.transform(Type.LIST.convert(obj, what),
        new com.google.common.base.Function<Object, TYPE>() {
          @Override
          public TYPE apply(Object input) {
            try {
              return type.cast(input);
            } catch (ClassCastException e) {
              throw new IllegalArgumentException(String.format(
                  "expected %s type for '%s' but got %s instead",
                  type.getSimpleName(), what, EvalUtils.getDatatypeName(input)));
            }
          }
    });
  }

  public static <KEY_TYPE, VALUE_TYPE> Iterable<Map.Entry<KEY_TYPE, VALUE_TYPE>> castMap(Object obj,
      final Class<KEY_TYPE> keyType, final Class<VALUE_TYPE> valueType, final String what) {
    if (obj == null) {
      return ImmutableList.of();
    }
    if (!(obj instanceof Map<?, ?>)) {
      throw new IllegalArgumentException(String.format(
          "expected a dictionary for %s but got %s instead",
          what, EvalUtils.getDatatypeName(obj)));
    }
    return Iterables.transform(((Map<?, ?>) obj).entrySet(),
        new com.google.common.base.Function<Map.Entry<?, ?>, Map.Entry<KEY_TYPE, VALUE_TYPE>>() {
          // This is safe. We check the type of the key-value pairs for every entry in the Map.
          // In Map.Entry the key always has the type of the first generic parameter, the
          // value has the second.
          @SuppressWarnings("unchecked")
            @Override
            public Map.Entry<KEY_TYPE, VALUE_TYPE> apply(Map.Entry<?, ?> input) {
            if (keyType.isAssignableFrom(input.getKey().getClass())) {
              if (input.getValue() == Environment.NONE) {
                input.setValue(null);
                return (Map.Entry<KEY_TYPE, VALUE_TYPE>) input;
              } else if (valueType.isAssignableFrom(input.getValue().getClass())) {
                return (Map.Entry<KEY_TYPE, VALUE_TYPE>) input;
              }
            }
            throw new IllegalArgumentException(String.format(
                "expected <%s, %s> type for '%s' but got <%s, %s> instead",
                keyType.getSimpleName(), valueType.getSimpleName(), what,
                EvalUtils.getDatatypeName(input.getKey()),
                EvalUtils.getDatatypeName(input.getValue())));
          }
        });
  }

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

  public static <TYPE> TYPE cast(Object elem, Class<TYPE> type, String what, Location loc)
      throws EvalException {
    try {
      return type.cast(elem);
    } catch (ClassCastException e) {
      throw new EvalException(loc, String.format("expected %s for '%s' but got %s instead",
          type.getSimpleName(), what, EvalUtils.getDatatypeName(elem)));
    }
  }

  @VisibleForTesting
  public static Iterable<Function> getStaticBuiltInFunctions() {
    return ImmutableList.of((Function) fileType);
  }
}
