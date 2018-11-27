// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.skylark;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.Attribute.ImmutableAttributeFactory;
import com.google.devtools.build.lib.packages.Attribute.SkylarkComputedDefaultTemplate;
import com.google.devtools.build.lib.packages.Attribute.SplitTransitionProvider;
import com.google.devtools.build.lib.packages.AttributeValueSource;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.SkylarkAspect;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAttrApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkCallbackFunction;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.SkylarkUtils;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.syntax.Type.LabelClass;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A helper class to provide Attr module in Skylark.
 *
 * <p>It exposes functions (for example, 'attr.string', 'attr.label_list', etc.) to Skylark users.
 * The functions are executed through reflection. As everywhere in Skylark, arguments are
 * type-checked with the signature and cannot be null.
 */
public final class SkylarkAttr implements SkylarkAttrApi {

  // Arguments

  private static boolean containsNonNoneKey(SkylarkDict<String, Object> arguments, String key) {
    return arguments.containsKey(key) && arguments.get(key) != Runtime.NONE;
  }

  private static void setAllowedFileTypes(
      String attr, Object fileTypesObj, FuncallExpression ast, Attribute.Builder<?> builder)
      throws EvalException {
    if (fileTypesObj == Boolean.TRUE) {
      builder.allowedFileTypes(FileTypeSet.ANY_FILE);
    } else if (fileTypesObj == Boolean.FALSE) {
      builder.allowedFileTypes(FileTypeSet.NO_FILE);
    } else if (fileTypesObj instanceof SkylarkFileType) {
      // TODO(laurentlb): deprecated, to be removed
      builder.allowedFileTypes(((SkylarkFileType) fileTypesObj).getFileTypeSet());
    } else if (fileTypesObj instanceof SkylarkList) {
      List<String> arg =
          SkylarkList.castSkylarkListOrNoneToList(
              fileTypesObj, String.class, "allow_files argument");
      builder.allowedFileTypes(FileType.of(arg));
    } else {
      throw new EvalException(
          ast.getLocation(), attr + " should be a boolean or a string list");
    }
  }

  private static ImmutableAttributeFactory createAttributeFactory(
      Type<?> type, SkylarkDict<String, Object> arguments, FuncallExpression ast, Environment env)
      throws EvalException {
    // We use an empty name now so that we can set it later.
    // This trick makes sense only in the context of Skylark (builtin rules should not use it).
    return createAttributeFactory(type, arguments, ast, env, "");
  }

  private static ImmutableAttributeFactory createAttributeFactory(
      Type<?> type,
      SkylarkDict<String, Object> arguments,
      FuncallExpression ast,
      Environment env,
      String name)
      throws EvalException {
    return createAttribute(type, arguments, ast, env, name).buildPartial();
  }

  private static Attribute.Builder<?> createAttribute(
      Type<?> type,
      SkylarkDict<String, Object> arguments,
      FuncallExpression ast,
      Environment env,
      String name)
      throws EvalException {
    Attribute.Builder<?> builder = Attribute.attr(name, type);

    Object defaultValue = arguments.get(DEFAULT_ARG);
    if (!EvalUtils.isNullOrNone(defaultValue)) {
      if (defaultValue instanceof UserDefinedFunction) {
        // Computed attribute. Non label type attributes already caused a type check error.
        SkylarkCallbackFunction callback =
            new SkylarkCallbackFunction(
                (UserDefinedFunction) defaultValue, ast, env.getSemantics());
        // SkylarkComputedDefaultTemplate needs to know the names of all attributes that it depends
        // on. However, this method does not know anything about other attributes.
        // We solve this problem by asking the SkylarkCallbackFunction for the parameter names used
        // in the function definition, which must be the names of attributes used by the callback.
        builder.value(
            new SkylarkComputedDefaultTemplate(
                type, callback.getParameterNames(), callback, ast.getLocation()));
      } else if (defaultValue instanceof SkylarkLateBoundDefault) {
        builder.value((SkylarkLateBoundDefault) defaultValue);
      } else {
        builder.defaultValue(defaultValue, env.getGlobals().getLabel(), DEFAULT_ARG);
      }
    }

    for (String flag : SkylarkList.castSkylarkListOrNoneToList(
        arguments.get(FLAGS_ARG), String.class, FLAGS_ARG)) {
      builder.setPropertyFlag(flag);
    }

    if (containsNonNoneKey(arguments, MANDATORY_ARG) && (Boolean) arguments.get(MANDATORY_ARG)) {
      builder.setPropertyFlag("MANDATORY");
    }

    if (containsNonNoneKey(arguments, NON_EMPTY_ARG)
        && (Boolean) arguments.get(NON_EMPTY_ARG)) {
      if (env.getSemantics().incompatibleDisableDeprecatedAttrParams()) {
        throw new EvalException(
            ast.getLocation(),
            "'non_empty' is no longer supported. use allow_empty instead. You can use "
                + "--incompatible_disable_deprecated_attr_params=false to temporarily disable this "
                + "check.");
      }

      builder.setPropertyFlag("NON_EMPTY");
    }

    if (containsNonNoneKey(arguments, ALLOW_EMPTY_ARG)
        && !(Boolean) arguments.get(ALLOW_EMPTY_ARG)) {
      builder.setPropertyFlag("NON_EMPTY");
    }

    if (containsNonNoneKey(arguments, EXECUTABLE_ARG) && (Boolean) arguments.get(EXECUTABLE_ARG)) {
      builder.setPropertyFlag("EXECUTABLE");
      if (!containsNonNoneKey(arguments, CONFIGURATION_ARG)) {
        throw new EvalException(
            ast.getLocation(),
            "cfg parameter is mandatory when executable=True is provided. Please see "
            + "https://www.bazel.build/versions/master/docs/skylark/rules.html#configurations "
            + "for more details.");
      }
    }

    if (containsNonNoneKey(arguments, SINGLE_FILE_ARG)
        && (Boolean) arguments.get(SINGLE_FILE_ARG)) {
      if (env.getSemantics().incompatibleDisableDeprecatedAttrParams()) {
        throw new EvalException(
            ast.getLocation(),
            "'single_file' is no longer supported. use allow_single_file instead. You can use "
                + "--incompatible_disable_deprecated_attr_params=false to temporarily disable this "
                + "check.");
      }
      if (containsNonNoneKey(arguments, ALLOW_SINGLE_FILE_ARG)) {
        throw new EvalException(
            ast.getLocation(),
            "Cannot specify both single_file (deprecated) and allow_single_file");
      }

      builder.setPropertyFlag("SINGLE_ARTIFACT");
    }

    if (containsNonNoneKey(arguments, ALLOW_FILES_ARG)
        && containsNonNoneKey(arguments, ALLOW_SINGLE_FILE_ARG)) {
      throw new EvalException(
          ast.getLocation(), "Cannot specify both allow_files and allow_single_file");
    }

    if (containsNonNoneKey(arguments, ALLOW_FILES_ARG)) {
      Object fileTypesObj = arguments.get(ALLOW_FILES_ARG);
      setAllowedFileTypes(ALLOW_FILES_ARG, fileTypesObj, ast, builder);
    } else if (containsNonNoneKey(arguments, ALLOW_SINGLE_FILE_ARG)) {
      Object fileTypesObj = arguments.get(ALLOW_SINGLE_FILE_ARG);
      setAllowedFileTypes(ALLOW_SINGLE_FILE_ARG, fileTypesObj, ast, builder);
      builder.setPropertyFlag("SINGLE_ARTIFACT");
    } else if (type.getLabelClass() == LabelClass.DEPENDENCY) {
      builder.allowedFileTypes(FileTypeSet.NO_FILE);
    }

    Object ruleClassesObj = arguments.get(ALLOW_RULES_ARG);
    if (ruleClassesObj != null && ruleClassesObj != Runtime.NONE) {
      builder.allowedRuleClasses(
          SkylarkList.castSkylarkListOrNoneToList(
              ruleClassesObj, String.class, "allowed rule classes for attribute definition"));
    }

    List<Object> values = SkylarkList.castSkylarkListOrNoneToList(
        arguments.get(VALUES_ARG), Object.class, VALUES_ARG);
    if (!Iterables.isEmpty(values)) {
      builder.allowedValues(new AllowedValueSet(values));
    }

    if (containsNonNoneKey(arguments, PROVIDERS_ARG)) {
      Object obj = arguments.get(PROVIDERS_ARG);
      SkylarkType.checkType(obj, SkylarkList.class, PROVIDERS_ARG);
      ImmutableList<ImmutableSet<SkylarkProviderIdentifier>> providersList = buildProviderPredicate(
          (SkylarkList<?>) obj, PROVIDERS_ARG, ast.getLocation());

      // If there is at least one empty set, there is no restriction.
      if (providersList.stream().noneMatch(ImmutableSet::isEmpty)) {
        builder.mandatoryProvidersList(providersList);
      }
    }

    if (containsNonNoneKey(arguments, CONFIGURATION_ARG)) {
      Object trans = arguments.get(CONFIGURATION_ARG);
      if (trans.equals("data")) {
        // This used to apply the "disable LIPO" (a.k.a. "data") transition. But now that LIPO is
        // turned down this is a noop. Still, there are cfg = "data"' references in the depot. So
        // we have to remove them via b/28688645 before we can remove this path.
        if (env.getSemantics().incompatibleDisallowDataTransition()) {
          throw new EvalException(ast.getLocation(),
              "Using cfg = \"data\" on an attribute is a noop and no longer supported. Please "
                  + "remove it. You can use --incompatible_disallow_data_transition=false to "
                  + "temporarily disable this check.");
        }
      } else if (trans.equals("host")) {
        builder.cfg(HostTransition.INSTANCE);
      } else if (trans instanceof SplitTransition) {
        builder.cfg((SplitTransition) trans);
      } else if (trans instanceof SplitTransitionProvider) {
        builder.cfg((SplitTransitionProvider) trans);
      } else if (trans instanceof StarlarkDefinedConfigTransition) {
        StarlarkDefinedConfigTransition starlarkDefinedTransition =
            (StarlarkDefinedConfigTransition) trans;
        if (starlarkDefinedTransition.isForAnalysisTesting()) {
          builder.hasAnalysisTestTransition();
        } else {
          builder.hasStarlarkDefinedTransition();
        }
        builder.cfg(new FunctionSplitTransitionProvider(starlarkDefinedTransition));
      } else if (!trans.equals("target")) {
        throw new EvalException(ast.getLocation(),
            "cfg must be either 'data', 'host', or 'target'.");
      }
    }

    if (containsNonNoneKey(arguments, ASPECTS_ARG)) {
      Object obj = arguments.get(ASPECTS_ARG);
      SkylarkType.checkType(obj, SkylarkList.class, ASPECTS_ARG);

      List<SkylarkAspect> aspects =
          ((SkylarkList<?>) obj).getContents(SkylarkAspect.class, "aspects");
      for (SkylarkAspect aspect : aspects) {
        aspect.attachToAttribute(builder, ast.getLocation());
      }
    }

    return builder;
  }

  /**
   * Builds a list of sets of accepted providers from Skylark list {@code obj}.
   * The list can either be a list of providers (in that case the result is a list with one
   * set) or a list of lists of providers (then the result is the list of sets).
   * @param argumentName used in error messages.
   * @param location location for error messages.
   */
  static ImmutableList<ImmutableSet<SkylarkProviderIdentifier>> buildProviderPredicate(
      SkylarkList<?> obj, String argumentName, Location location) throws EvalException {
    if (obj.isEmpty()) {
      return ImmutableList.of();
    }
    boolean isListOfProviders = true;
    for (Object o : obj) {
      if (!isProvider(o)) {
        isListOfProviders = false;
        break;
      }
    }
    if (isListOfProviders) {
      return ImmutableList.of(getSkylarkProviderIdentifiers(obj, location));
    } else {
      return getProvidersList(obj, argumentName, location);
    }
  }

  /**
   * Returns true if {@code o} is a Skylark provider (either a declared provider or
   * a legacy provider name.
   */
  static boolean isProvider(Object o) {
    return o instanceof String || o instanceof Provider;
  }

  /**
   * Converts Skylark identifiers of providers (either a string or a provider value)
   * to their internal representations.
   */
  static ImmutableSet<SkylarkProviderIdentifier> getSkylarkProviderIdentifiers(
      SkylarkList<?> list, Location location) throws EvalException {
    ImmutableList.Builder<SkylarkProviderIdentifier> result = ImmutableList.builder();

    for (Object obj : list) {
      if (obj instanceof String) {
        result.add(SkylarkProviderIdentifier.forLegacy((String) obj));
      } else if (obj instanceof Provider) {
        Provider constructor = (Provider) obj;
        if (!constructor.isExported()) {
          throw new EvalException(location,
              "Providers should be top-level values in extension files that define them.");
        }
        result.add(SkylarkProviderIdentifier.forKey(constructor.getKey()));
      }
    }
    return ImmutableSet.copyOf(result.build());
  }

  private static ImmutableList<ImmutableSet<SkylarkProviderIdentifier>> getProvidersList(
      SkylarkList<?> skylarkList, String argumentName, Location location) throws EvalException {
    ImmutableList.Builder<ImmutableSet<SkylarkProviderIdentifier>> providersList =
        ImmutableList.builder();
    String errorMsg = "Illegal argument: element in '%s' is of unexpected type. "
        + "Either all elements should be providers, "
        + "or all elements should be lists of providers, but got %s.";

    for (Object o : skylarkList) {
      if (!(o instanceof SkylarkList)) {
        throw new EvalException(location, String.format(errorMsg, PROVIDERS_ARG,
            "an element of type " + EvalUtils.getDataTypeName(o, true)));
      }
      for (Object value : (SkylarkList) o) {
        if (!isProvider(value)) {
          throw new EvalException(location, String.format(errorMsg, argumentName,
              "list with an element of type "
                  + EvalUtils.getDataTypeNameFromClass(value.getClass())));
        }
      }
      providersList.add(getSkylarkProviderIdentifiers((SkylarkList<?>) o, location));
    }
    return providersList.build();
  }

  private static Descriptor createAttrDescriptor(
      String name,
      SkylarkDict<String, Object> kwargs,
      Type<?> type,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    try {
      return new Descriptor(name, createAttributeFactory(type, kwargs, ast, env));
    } catch (ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  private static final Map<Type<?>, String> whyNotConfigurable =
      ImmutableMap.<Type<?>, String>builder()
          .put(BuildType.LICENSE,
              "loading phase license checking logic assumes non-configurable values")
          .put(BuildType.OUTPUT, "output paths are part of the static graph structure")
          .build();

  /**
   * If the given attribute type is non-configurable, returns the reason why. Otherwise, returns
   * {@code null}.
   */
  @Nullable
  public static String maybeGetNonConfigurableReason(Type<?> type) {
    return whyNotConfigurable.get(type);
  }

  private static Descriptor createNonconfigurableAttrDescriptor(
      String name,
      SkylarkDict<String, Object> kwargs,
      Type<?> type,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    String whyNotConfigurableReason =
        Preconditions.checkNotNull(maybeGetNonConfigurableReason(type), type);
    try {
      // We use an empty name now so that we can set it later.
      // This trick makes sense only in the context of Skylark (builtin rules should not use it).
      return new Descriptor(
          name,
          createAttribute(type, kwargs, ast, env, "")
              .nonconfigurable(whyNotConfigurableReason)
              .buildPartial());
    } catch (ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<attr>");
  }

  @Override
  public Descriptor intAttribute(
      Integer defaultInt,
      String doc,
      Boolean mandatory,
      SkylarkList<?> values,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    // TODO(bazel-team): Replace literal strings with constants.
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.int", ast.getLocation());
    return createAttrDescriptor(
        "int",
        EvalUtils.<String, Object>optionMap(
            env, DEFAULT_ARG, defaultInt, MANDATORY_ARG, mandatory, VALUES_ARG, values),
        Type.INTEGER,
        ast,
        env);
  }

  @Override
  public Descriptor stringAttribute(
      String defaultString,
      String doc,
      Boolean mandatory,
      SkylarkList<?> values,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.string", ast.getLocation());
    return createAttrDescriptor(
        "string",
        EvalUtils.<String, Object>optionMap(
            env, DEFAULT_ARG, defaultString, MANDATORY_ARG, mandatory, VALUES_ARG, values),
        Type.STRING,
        ast,
        env);
  }

  @Override
  public Descriptor labelAttribute(
      Object defaultO,
      String doc,
      Boolean executable,
      Object allowFiles,
      Object allowSingleFile,
      Boolean mandatory,
      SkylarkList<?> providers,
      Object allowRules,
      Boolean singleFile,
      Object cfg,
      SkylarkList<?> aspects,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.label", ast.getLocation());
    try {
      ImmutableAttributeFactory attribute =
          createAttributeFactory(
              BuildType.LABEL,
              EvalUtils.<String, Object>optionMap(
                  env,
                  DEFAULT_ARG,
                  defaultO,
                  EXECUTABLE_ARG,
                  executable,
                  ALLOW_FILES_ARG,
                  allowFiles,
                  ALLOW_SINGLE_FILE_ARG,
                  allowSingleFile,
                  MANDATORY_ARG,
                  mandatory,
                  PROVIDERS_ARG,
                  providers,
                  ALLOW_RULES_ARG,
                  allowRules,
                  SINGLE_FILE_ARG,
                  singleFile,
                  CONFIGURATION_ARG,
                  cfg,
                  ASPECTS_ARG,
                  aspects),
              ast,
              env,
              "label");
      return new Descriptor("label", attribute);
    } catch (EvalException e) {
      throw new EvalException(ast.getLocation(), e.getMessage(), e);
    }
  }

  @Override
  public Descriptor stringListAttribute(
      Boolean mandatory,
      Boolean nonEmpty,
      Boolean allowEmpty,
      SkylarkList<?> defaultList,
      String doc,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.string_list", ast.getLocation());
    return createAttrDescriptor(
        "string_list",
        EvalUtils.<String, Object>optionMap(
            env,
            DEFAULT_ARG,
            defaultList,
            MANDATORY_ARG,
            mandatory,
            NON_EMPTY_ARG,
            nonEmpty,
            ALLOW_EMPTY_ARG,
            allowEmpty),
        Type.STRING_LIST,
        ast,
        env);
  }

  @Override
  public Descriptor intListAttribute(
      Boolean mandatory,
      Boolean nonEmpty,
      Boolean allowEmpty,
      SkylarkList<?> defaultList,
      String doc,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.int_list", ast.getLocation());
    return createAttrDescriptor(
        "int_list",
        EvalUtils.<String, Object>optionMap(
            env,
            DEFAULT_ARG,
            defaultList,
            MANDATORY_ARG,
            mandatory,
            NON_EMPTY_ARG,
            nonEmpty,
            ALLOW_EMPTY_ARG,
            allowEmpty),
        Type.INTEGER_LIST,
        ast,
        env);
  }

  @Override
  public Descriptor labelListAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
      Object allowFiles,
      Object allowRules,
      SkylarkList<?> providers,
      SkylarkList<?> flags,
      Boolean mandatory,
      Boolean nonEmpty,
      Object cfg,
      SkylarkList<?> aspects,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.label_list", ast.getLocation());
    SkylarkDict<String, Object> kwargs =
        EvalUtils.<String, Object>optionMap(
            env,
            DEFAULT_ARG,
            defaultList,
            ALLOW_FILES_ARG,
            allowFiles,
            ALLOW_RULES_ARG,
            allowRules,
            PROVIDERS_ARG,
            providers,
            FLAGS_ARG,
            flags,
            MANDATORY_ARG,
            mandatory,
            NON_EMPTY_ARG,
            nonEmpty,
            ALLOW_EMPTY_ARG,
            allowEmpty,
            CONFIGURATION_ARG,
            cfg,
            ASPECTS_ARG,
            aspects);
    try {
      ImmutableAttributeFactory attribute =
          createAttributeFactory(BuildType.LABEL_LIST, kwargs, ast, env, "label_list");
      return new Descriptor("label_list", attribute);
    } catch (EvalException e) {
      throw new EvalException(ast.getLocation(), e.getMessage(), e);
    }
  }

  @Override
  public Descriptor labelKeyedStringDictAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
      Object allowFiles,
      Object allowRules,
      SkylarkList<?> providers,
      SkylarkList<?> flags,
      Boolean mandatory,
      Boolean nonEmpty,
      Object cfg,
      SkylarkList<?> aspects,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(
        env, "attr.label_keyed_string_dict", ast.getLocation());
    SkylarkDict<String, Object> kwargs =
        EvalUtils.<String, Object>optionMap(
            env,
            DEFAULT_ARG,
            defaultList,
            ALLOW_FILES_ARG,
            allowFiles,
            ALLOW_RULES_ARG,
            allowRules,
            PROVIDERS_ARG,
            providers,
            FLAGS_ARG,
            flags,
            MANDATORY_ARG,
            mandatory,
            NON_EMPTY_ARG,
            nonEmpty,
            ALLOW_EMPTY_ARG,
            allowEmpty,
            CONFIGURATION_ARG,
            cfg,
            ASPECTS_ARG,
            aspects);
    try {
      ImmutableAttributeFactory attribute =
          createAttributeFactory(
              BuildType.LABEL_KEYED_STRING_DICT, kwargs, ast, env, "label_keyed_string_dict");
      return new Descriptor("label_keyed_string_dict", attribute);
    } catch (EvalException e) {
      throw new EvalException(ast.getLocation(), e.getMessage(), e);
    }
  }

  @Override
  public Descriptor boolAttribute(
      Boolean defaultO, String doc, Boolean mandatory, FuncallExpression ast, Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.bool", ast.getLocation());
    return createAttrDescriptor(
        "bool",
        EvalUtils.<String, Object>optionMap(env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory),
        Type.BOOLEAN,
        ast,
        env);
  }

  @Override
  public Descriptor outputAttribute(
      Object defaultO, String doc, Boolean mandatory, FuncallExpression ast, Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.output", ast.getLocation());

    return createNonconfigurableAttrDescriptor(
        "output",
        EvalUtils.<String, Object>optionMap(env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory),
        BuildType.OUTPUT,
        ast,
        env);
  }

  @Override
  public Descriptor outputListAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
      Boolean mandatory,
      Boolean nonEmpty,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.output_list", ast.getLocation());

    return createAttrDescriptor(
        "output_list",
        EvalUtils.<String, Object>optionMap(
            env,
            DEFAULT_ARG,
            defaultList,
            MANDATORY_ARG,
            mandatory,
            NON_EMPTY_ARG,
            nonEmpty,
            ALLOW_EMPTY_ARG,
            allowEmpty),
        BuildType.OUTPUT_LIST,
        ast,
        env);
  }

  @Override
  public Descriptor stringDictAttribute(
      Boolean allowEmpty,
      SkylarkDict<?, ?> defaultO,
      String doc,
      Boolean mandatory,
      Boolean nonEmpty,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.string_dict", ast.getLocation());
    return createAttrDescriptor(
        "string_dict",
        EvalUtils.<String, Object>optionMap(
            env,
            DEFAULT_ARG,
            defaultO,
            MANDATORY_ARG,
            mandatory,
            NON_EMPTY_ARG,
            nonEmpty,
            ALLOW_EMPTY_ARG,
            allowEmpty),
        Type.STRING_DICT,
        ast,
        env);
  }

  @Override
  public Descriptor stringListDictAttribute(
      Boolean allowEmpty,
      SkylarkDict<?, ?> defaultO,
      String doc,
      Boolean mandatory,
      Boolean nonEmpty,
      FuncallExpression ast,
      Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.string_list_dict", ast.getLocation());
    return createAttrDescriptor(
        "string_list_dict",
        EvalUtils.<String, Object>optionMap(
            env,
            DEFAULT_ARG,
            defaultO,
            MANDATORY_ARG,
            mandatory,
            NON_EMPTY_ARG,
            nonEmpty,
            ALLOW_EMPTY_ARG,
            allowEmpty),
        Type.STRING_LIST_DICT,
        ast,
        env);
  }

  @Override
  public Descriptor licenseAttribute(
      Object defaultO, String doc, Boolean mandatory, FuncallExpression ast, Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "attr.license", ast.getLocation());
    return createNonconfigurableAttrDescriptor(
        "license",
        EvalUtils.<String, Object>optionMap(env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory),
        BuildType.LICENSE,
        ast,
        env);
  }

  /** A descriptor of an attribute defined in Skylark. */
  @AutoCodec
  public static final class Descriptor implements SkylarkAttrApi.Descriptor {
    private final ImmutableAttributeFactory attributeFactory;
    private final String name;

    @AutoCodec.VisibleForSerialization
    Descriptor(String name, ImmutableAttributeFactory attributeFactory) {
      this.attributeFactory = Preconditions.checkNotNull(attributeFactory);
      this.name = name;
    }

    public boolean hasDefault() {
      return attributeFactory.isValueSet();
    }

    public AttributeValueSource getValueSource() {
      return attributeFactory.getValueSource();
    }

    public Attribute build(String name) {
      return attributeFactory.build(name);
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("<attr." + name + ">");
    }
  }
}
