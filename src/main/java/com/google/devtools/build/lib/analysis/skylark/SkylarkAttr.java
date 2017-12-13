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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.SkylarkComputedDefaultTemplate;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.SplitTransitionProvider;
import com.google.devtools.build.lib.packages.AttributeValueSource;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.SkylarkAspect;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkCallbackFunction;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
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
 * <p>It exposes functions (e.g. 'attr.string', 'attr.label_list', etc.) to Skylark users. The
 * functions are executed through reflection. As everywhere in Skylark, arguments are type-checked
 * with the signature and cannot be null.
 */
@SkylarkModule(
  name = "attr",
  namespace = true,
  category = SkylarkModuleCategory.BUILTIN,
  doc =
      "Module for creating new attributes. "
          + "They are only for use with <a href=\"globals.html#rule\">rule</a> or "
          + "<a href=\"globals.html#aspect\">aspect</a>."
)
public final class SkylarkAttr implements SkylarkValue {

  // Arguments

  private static final String ALLOW_FILES_ARG = "allow_files";
  private static final String ALLOW_FILES_DOC =
      "whether File targets are allowed. Can be True, False (default), or a list of file "
      + "extensions that are allowed (e.g. <code>[\".cc\", \".cpp\"]</code>).";

  private static final String ALLOW_RULES_ARG = "allow_rules";
  private static final String ALLOW_RULES_DOC =
      "which rule targets (name of the classes) are allowed. This is deprecated (kept only for "
          + "compatibility), use providers instead.";

  private static final String ASPECTS_ARG = "aspects";
  private static final String ASPECTS_ARG_DOC =
      "aspects that should be applied to the dependency or dependencies specified by this "
          + "attribute";

  private static final String CONFIGURATION_ARG = "cfg";
  private static final String CONFIGURATION_DOC =
      "configuration of the attribute. It can be either \"data\", \"host\", or \"target\". "
          + "This parameter is required if <code>executable</code> is True.";

  private static final String DEFAULT_ARG = "default";
  private static final String DEFAULT_DOC = "the default value of the attribute.";

  private static final String DOC_ARG = "doc";
  private static final String DOC_DOC =
      "a description of the attribute that can be extracted by documentation generating tools.";

  private static final String EXECUTABLE_ARG = "executable";
  private static final String EXECUTABLE_DOC =
      "True if the labels have to be executable. This means the label must refer to an "
          + "executable file, or to a rule that outputs an executable file. Access the labels "
          + "with <code>ctx.executable.&lt;attribute_name&gt;</code>.";

  private static final String FLAGS_ARG = "flags";
  private static final String FLAGS_DOC = "deprecated, will be removed";

  private static final String MANDATORY_ARG = "mandatory";
  private static final String MANDATORY_DOC = "True if the value must be explicitly specified";

  private static final String NON_EMPTY_ARG = "non_empty";
  private static final String NON_EMPTY_DOC =
      "True if the attribute must not be empty. Deprecated: Use allow_empty instead.";

  private static final String ALLOW_EMPTY_ARG = "allow_empty";
  private static final String ALLOW_EMPTY_DOC = "True if the attribute can be empty";

  private static final String PROVIDERS_ARG = "providers";
  private static final String PROVIDERS_DOC =
      "mandatory providers list. It should be either a list of providers, or a "
          + "list of lists of providers. Every dependency should provide ALL providers "
          + "from at least ONE of these lists. A single list of providers will be "
          + "automatically converted to a list containing one list of providers.";

  private static final String SINGLE_FILE_ARG = "single_file";
  private static final String ALLOW_SINGLE_FILE_ARG = "allow_single_file";

  private static final String VALUES_ARG = "values";
  private static final String VALUES_DOC =
      "the list of allowed values for the attribute. An error is raised if any other "
          + "value is given.";

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

  private static Attribute.Builder<?> createAttribute(
      Type<?> type, SkylarkDict<String, Object> arguments, FuncallExpression ast, Environment env)
      throws EvalException {
    // We use an empty name now so that we can set it later.
    // This trick makes sense only in the context of Skylark (builtin rules should not use it).
    return createAttribute(type, arguments, ast, env, "");
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
        builder.defaultValue(defaultValue, env.getGlobals().getTransitiveLabel(), DEFAULT_ARG);
      }
    }

    for (String flag : SkylarkList.castSkylarkListOrNoneToList(
        arguments.get(FLAGS_ARG), String.class, FLAGS_ARG)) {
      builder.setPropertyFlag(flag);
    }

    if (containsNonNoneKey(arguments, MANDATORY_ARG) && (Boolean) arguments.get(MANDATORY_ARG)) {
      builder.setPropertyFlag("MANDATORY");
    }

    // TODO(laurentlb): Deprecated, remove in August 2016 (use allow_empty instead).
    if (containsNonNoneKey(arguments, NON_EMPTY_ARG) && (Boolean) arguments.get(NON_EMPTY_ARG)) {
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

    // TODO(laurentlb): Deprecated, remove in August 2016 (use allow_single_file).
    if (containsNonNoneKey(arguments, SINGLE_FILE_ARG)
        && (Boolean) arguments.get(SINGLE_FILE_ARG)) {
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
        builder.cfg(ConfigurationTransition.DATA);
      } else if (trans.equals("host")) {
        builder.cfg(ConfigurationTransition.HOST);
      } else if (trans instanceof SplitTransition<?>) {
        builder.cfg((SplitTransition<?>) trans);
      } else if (trans instanceof SplitTransitionProvider) {
        builder.cfg((SplitTransitionProvider) trans);
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
        if (!aspect.isExported()) {
          throw new EvalException(
              ast.getLocation(),
              "Aspects should be top-level values in extension files that define them.");
        }
        builder.aspect(aspect, ast.getLocation());
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
      return new Descriptor(name, createAttribute(type, kwargs, ast, env));
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
      return new Descriptor(
          name, createAttribute(type, kwargs, ast, env).nonconfigurable(whyNotConfigurableReason));
    } catch (ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<attr>");
  }

  @SkylarkSignature(
    name = "int",
    doc = "Creates an attribute of type int.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = Integer.class,
        defaultValue = "0",
        doc = DEFAULT_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        doc = MANDATORY_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = VALUES_ARG,
        type = SkylarkList.class,
        generic1 = Integer.class,
        defaultValue = "[]",
        doc = VALUES_DOC,
        named = true,
        positional = false
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction integer =
      new BuiltinFunction("int") {
        public Descriptor invoke(
            Integer defaultInt,
            String doc,
            Boolean mandatory,
            SkylarkList<?> values,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          // TODO(bazel-team): Replace literal strings with constants.
          env.checkLoadingOrWorkspacePhase("attr.int", ast.getLocation());
          return createAttrDescriptor(
              getName(),
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultInt, MANDATORY_ARG, mandatory, VALUES_ARG, values),
              Type.INTEGER,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "configuration_field",
    returnType = SkylarkLateBoundDefault.class,
    // TODO(cparsons): Provide a link to documentation for available SkylarkConfigurationFields.
    doc = "References a late-bound default value for an attribute of type "
      + "<a href=\"attr.html#label\">label</a>. A value is 'late-bound' if it requires "
      + "the configuration to be built before determining the value. Any attribute using this "
      + "as a value must <a href=\"../rules.html#private-attributes\">be private</a>.",
    parameters = {
        @Param(
            name = "fragment",
            type = String.class,
            doc = "The name of a configuration fragment which contains the late-bound value."
        ),
        @Param(
            name = "name",
            type = String.class,
            doc = "The name of the value to obtain from the configuration fragment."),
    },
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction configurationField =
      new BuiltinFunction("configuration_field") {
        public SkylarkLateBoundDefault<?> invoke(
            String fragment, String name, Location loc, Environment env)
            throws EvalException {
          Class<?> fragmentClass = SkylarkUtils.getFragmentMap(env).get(fragment);

          if (fragmentClass == null) {
            throw new EvalException(
                loc,
                String.format("invalid configuration fragment name '%s'", fragment));
          }
          try {
            return SkylarkLateBoundDefault.forConfigurationField(
                fragmentClass, name, SkylarkUtils.getToolsRepository(env));
          } catch (SkylarkLateBoundDefault.InvalidConfigurationFieldException exception) {
            throw new EvalException(loc, exception);
          }
        }
      };

  @SkylarkSignature(
    name = "string",
    doc = "Creates an attribute of type <a href=\"string.html\">string</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DEFAULT_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        doc = MANDATORY_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = VALUES_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        doc = VALUES_DOC,
        named = true,
        positional = false
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction string =
      new BuiltinFunction("string") {
        public Descriptor invoke(
            String defaultString,
            String doc,
            Boolean mandatory,
            SkylarkList<?> values,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.string", ast.getLocation());
          return createAttrDescriptor(
              getName(),
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultString, MANDATORY_ARG, mandatory, VALUES_ARG, values),
              Type.STRING,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "label",
    doc =
        "Creates an attribute of type <a href=\"Target.html\">Target</a> which is the target "
            + "referred to by the label. "
            + "It is the only way to specify a dependency to another target. "
            + "If you need a dependency that the user cannot overwrite, "
            + "<a href=\"../rules.html#private-attributes\">make the attribute private</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        allowedTypes = {
          @ParamType(type = Label.class),
          @ParamType(type = String.class),
          @ParamType(type = SkylarkLateBoundDefault.class)
        },
        callbackEnabled = true,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            DEFAULT_DOC
                + "Use a string or the <a href=\"globals.html#Label\"><code>Label</code></a> "
                + "function to specify a default value ex: "
                + "<code>attr.label(default = \"//a:b\")</code>"
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = EXECUTABLE_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = EXECUTABLE_DOC
      ),
      @Param(
        name = ALLOW_FILES_ARG,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = ALLOW_FILES_DOC
      ),
      @Param(
        name = ALLOW_SINGLE_FILE_ARG,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "This is similar to <code>allow_files</code>, with the restriction that the label must "
                + "correspond to a single <a href=\"File.html\">File</a>. "
                + "Access it through <code>ctx.file.&lt;attribute_name&gt;</code>."
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = MANDATORY_DOC
      ),
      @Param(
        name = PROVIDERS_ARG,
        type = SkylarkList.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = PROVIDERS_DOC
      ),
      @Param(
        name = ALLOW_RULES_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = ALLOW_RULES_DOC
      ),
      @Param(
        name = SINGLE_FILE_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc =
            "Deprecated: Use <code>allow_single_file</code> instead. "
                + "If True, the label must correspond to a single <a href=\"File.html\">File</a>. "
                + "Access it through <code>ctx.file.&lt;attribute_name&gt;</code>."
      ),
      @Param(
        name = CONFIGURATION_ARG,
        type = Object.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = CONFIGURATION_DOC
      ),
      @Param(
        name = ASPECTS_ARG,
        type = SkylarkList.class,
        generic1 = SkylarkAspect.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = ASPECTS_ARG_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction label =
      new BuiltinFunction("label") {
        public Descriptor invoke(
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
          env.checkLoadingOrWorkspacePhase("attr.label", ast.getLocation());
          try {
            Attribute.Builder<?> attribute =
                createAttribute(
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
            return new Descriptor(getName(), attribute);
          } catch (EvalException e) {
            throw new EvalException(ast.getLocation(), e.getMessage(), e);
          }
        }
      };

  @SkylarkSignature(
    name = "string_list",
    doc =
        "Creates an attribute which is a <a href=\"list.html\">list</a> of "
            + "<a href=\"string.html\">strings</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        doc = DEFAULT_DOC
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        doc = MANDATORY_DOC
      ),
      @Param(
        name = NON_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        doc = NON_EMPTY_DOC
      ),
      @Param(
        name = ALLOW_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "True",
        doc = ALLOW_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction stringList =
      new BuiltinFunction("string_list") {
        public Descriptor invoke(
            SkylarkList<?> defaultList,
            String doc,
            Boolean mandatory,
            Boolean nonEmpty,
            Boolean allowEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.string_list", ast.getLocation());
          return createAttrDescriptor(
              getName(),
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
      };

  @SkylarkSignature(
    name = "int_list",
    doc = "Creates an attribute which is a <a href=\"list.html\">list</a> of ints.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkList.class,
        generic1 = Integer.class,
        defaultValue = "[]",
        doc = DEFAULT_DOC
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        doc = MANDATORY_DOC
      ),
      @Param(
        name = NON_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        doc = NON_EMPTY_DOC
      ),
      @Param(
        name = ALLOW_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "True",
        doc = ALLOW_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction intList =
      new BuiltinFunction("int_list") {
        public Descriptor invoke(
            SkylarkList<?> defaultList,
            String doc,
            Boolean mandatory,
            Boolean nonEmpty,
            Boolean allowEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.int_list", ast.getLocation());
          return createAttrDescriptor(
              getName(),
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
      };

  @SkylarkSignature(
    name = "label_list",
    doc =
        "Creates an attribute which is a <a href=\"list.html\">list</a> of type "
            + "<a href=\"Target.html\">Target</a> which are specified by the labels in the list. "
            + "See <a href=\"attr.html#label\">label</a> for more information.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkList.class,
        generic1 = Label.class,
        callbackEnabled = true,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc =
            DEFAULT_DOC
                + "Use strings or the <a href=\"globals.html#Label\"><code>Label</code></a> "
                + "function to specify default values ex: "
                + "<code>attr.label_list(default = [\"//a:b\", \"//a:c\"])</code>"
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = ALLOW_FILES_ARG, // bool or FileType filter
        defaultValue = "None",
        named = true,
        positional = false,
        doc = ALLOW_FILES_DOC
      ),
      @Param(
        name = ALLOW_RULES_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = ALLOW_RULES_DOC
      ),
      @Param(
        name = PROVIDERS_ARG,
        type = SkylarkList.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = PROVIDERS_DOC
      ),
      @Param(
        name = FLAGS_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = FLAGS_DOC
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = MANDATORY_DOC
      ),
      @Param(
        name = NON_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = NON_EMPTY_DOC
      ),
      @Param(
        name = ALLOW_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "True",
        doc = ALLOW_EMPTY_DOC
      ),
      @Param(
        name = CONFIGURATION_ARG,
        type = Object.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = CONFIGURATION_DOC
      ),
      @Param(
        name = ASPECTS_ARG,
        type = SkylarkList.class,
        generic1 = SkylarkAspect.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = ASPECTS_ARG_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction labelList =
      new BuiltinFunction("label_list") {
        public Descriptor invoke(
            Object defaultList,
            String doc,
            Object allowFiles,
            Object allowRules,
            SkylarkList<?> providers,
            SkylarkList<?> flags,
            Boolean mandatory,
            Boolean nonEmpty,
            Boolean allowEmpty,
            Object cfg,
            SkylarkList<?> aspects,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.label_list", ast.getLocation());
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
            Attribute.Builder<?> attribute =
                createAttribute(BuildType.LABEL_LIST, kwargs, ast, env, "label_list");
            return new Descriptor(getName(), attribute);
          } catch (EvalException e) {
            throw new EvalException(ast.getLocation(), e.getMessage(), e);
          }
        }
      };

  @SkylarkSignature(
    name = "label_keyed_string_dict",
    doc =
        "Creates an attribute which is a <a href=\"dict.html\">dict</a>. Its keys are type "
            + "<a href=\"Target.html\">Target</a> and are specified by the label keys of the "
            + "input dict. Its values are <a href=\"string.html\">strings</a>. See "
            + "<a href=\"attr.html#label\">label</a> for more information.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkDict.class,
        callbackEnabled = true,
        defaultValue = "{}",
        named = true,
        positional = false,
        doc =
            DEFAULT_DOC
                + "Use strings or the <a href=\"globals.html#Label\"><code>Label</code></a> "
                + "function to specify default values ex: "
                + "<code>attr.label_keyed_string_dict(default = "
                + "{\"//a:b\": \"value\", \"//a:c\": \"string\"})</code>"
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = ALLOW_FILES_ARG, // bool or FileType filter
        defaultValue = "None",
        named = true,
        positional = false,
        doc = ALLOW_FILES_DOC
      ),
      @Param(
        name = ALLOW_RULES_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = ALLOW_RULES_DOC
      ),
      @Param(
        name = PROVIDERS_ARG,
        type = SkylarkList.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = PROVIDERS_DOC
      ),
      @Param(
        name = FLAGS_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = FLAGS_DOC
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = MANDATORY_DOC
      ),
      @Param(
        name = NON_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = NON_EMPTY_DOC
      ),
      @Param(
        name = ALLOW_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "True",
        doc = ALLOW_EMPTY_DOC
      ),
      @Param(
        name = CONFIGURATION_ARG,
        type = Object.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = CONFIGURATION_DOC
      ),
      @Param(
        name = ASPECTS_ARG,
        type = SkylarkList.class,
        generic1 = SkylarkAspect.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = ASPECTS_ARG_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction labelKeyedStringDict =
      new BuiltinFunction("label_keyed_string_dict") {
        public Descriptor invoke(
            Object defaultList,
            String doc,
            Object allowFiles,
            Object allowRules,
            SkylarkList<?> providers,
            SkylarkList<?> flags,
            Boolean mandatory,
            Boolean nonEmpty,
            Boolean allowEmpty,
            Object cfg,
            SkylarkList<?> aspects,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.label_keyed_string_dict", ast.getLocation());
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
            Attribute.Builder<?> attribute =
                createAttribute(
                    BuildType.LABEL_KEYED_STRING_DICT, kwargs, ast, env, "label_keyed_string_dict");
            return new Descriptor(this.getName(), attribute);
          } catch (EvalException e) {
            throw new EvalException(ast.getLocation(), e.getMessage(), e);
          }
        }
      };

  @SkylarkSignature(
    name = "bool",
    doc = "Creates an attribute of type bool.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = DEFAULT_DOC
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = MANDATORY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction bool =
      new BuiltinFunction("bool") {
        public Descriptor invoke(
            Boolean defaultO, String doc, Boolean mandatory, FuncallExpression ast, Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.bool", ast.getLocation());
          return createAttrDescriptor(
              getName(),
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory),
              Type.BOOLEAN,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "output",
    doc =
        "Creates an attribute of type output. "
            + "The user provides a file name (string) and the rule must create an action that "
            + "generates the file.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = Label.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = DEFAULT_DOC
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = MANDATORY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction output =
      new BuiltinFunction("output") {
        public Descriptor invoke(
            Object defaultO, String doc, Boolean mandatory, FuncallExpression ast, Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.output", ast.getLocation());
          return createNonconfigurableAttrDescriptor(
              getName(),
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory),
              BuildType.OUTPUT,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "output_list",
    doc =
        "Creates an attribute which is a <a href=\"list.html\">list</a> of outputs. "
            + "See <a href=\"attr.html#output\">output</a> for more information.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkList.class,
        generic1 = Label.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = DEFAULT_DOC
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = MANDATORY_DOC
      ),
      @Param(
        name = NON_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = NON_EMPTY_DOC
      ),
      @Param(
        name = ALLOW_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "True",
        doc = ALLOW_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction outputList =
      new BuiltinFunction("output_list") {
        public Descriptor invoke(
            SkylarkList defaultList,
            String doc,
            Boolean mandatory,
            Boolean nonEmpty,
            Boolean allowEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.output_list", ast.getLocation());
          return createAttrDescriptor(
              getName(),
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
      };

  @SkylarkSignature(
    name = "string_dict",
    doc =
        "Creates an attribute of type <a href=\"dict.html\">dict</a>, mapping from "
            + "<a href=\"string.html\">string</a> to <a href=\"string.html\">string</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkDict.class,
        named = true,
        positional = false,
        defaultValue = "{}",
        doc = DEFAULT_DOC
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        named = true,
        positional = false,
        defaultValue = "False",
        doc = MANDATORY_DOC
      ),
      @Param(
        name = NON_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = NON_EMPTY_DOC
      ),
      @Param(
        name = ALLOW_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "True",
        doc = ALLOW_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction stringDict =
      new BuiltinFunction("string_dict") {
        public Descriptor invoke(
            SkylarkDict<?, ?> defaultO,
            String doc,
            Boolean mandatory,
            Boolean nonEmpty,
            Boolean allowEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.string_dict", ast.getLocation());
          return createAttrDescriptor(
              getName(),
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
      };

  @SkylarkSignature(
    name = "string_list_dict",
    doc =
        "Creates an attribute of type <a href=\"dict.html\">dict</a>, mapping from "
            + "<a href=\"string.html\">string</a> to <a href=\"list.html\">list</a> of "
            + "<a href=\"string.html\">string</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkDict.class,
        defaultValue = "{}",
        named = true,
        positional = false,
        doc = DEFAULT_DOC
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = MANDATORY_DOC
      ),
      @Param(
        name = NON_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = NON_EMPTY_DOC
      ),
      @Param(
        name = ALLOW_EMPTY_ARG,
        type = Boolean.class,
        defaultValue = "True",
        doc = ALLOW_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction stringListDict =
      new BuiltinFunction("string_list_dict") {
        public Descriptor invoke(
            SkylarkDict<?, ?> defaultO,
            String doc,
            Boolean mandatory,
            Boolean nonEmpty,
            Boolean allowEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.string_list_dict", ast.getLocation());
          return createAttrDescriptor(
              getName(),
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
      };

  @SkylarkSignature(
    name = "license",
    doc = "Creates an attribute of type license.",
    // TODO(bazel-team): Implement proper license support for Skylark.
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    parameters = {
      // TODO(bazel-team): ensure this is the correct default value
      @Param(
        name = DEFAULT_ARG,
        defaultValue = "None",
        noneable = true,
        named = true,
        positional = false,
        doc = DEFAULT_DOC
      ),
      @Param(
        name = DOC_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DOC_DOC,
        named = true,
        positional = false
      ),
      @Param(
        name = MANDATORY_ARG,
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = MANDATORY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction license =
      new BuiltinFunction("license") {
        public Descriptor invoke(
            Object defaultO, String doc, Boolean mandatory, FuncallExpression ast, Environment env)
            throws EvalException {
          env.checkLoadingOrWorkspacePhase("attr.license", ast.getLocation());
          return createNonconfigurableAttrDescriptor(
              getName(),
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory),
              BuildType.LICENSE,
              ast,
              env);
        }
      };

  /** A descriptor of an attribute defined in Skylark. */
  @SkylarkModule(
    name = "Attribute",
    category = SkylarkModuleCategory.NONE,
    doc =
        "Representation of a definition of an attribute. Use the <a href=\"attr\"> module to "
            + "create an Attribute. They are only for use with "
            + "<a href=\"globals.html#rule\">rule</a> or "
            + "<a href=\"globals.html#aspect\">aspect</a>."
  )
  public static final class Descriptor implements SkylarkValue {
    private final Attribute.Builder<?> attributeBuilder;

    /**
     * This lock guards {@code attributeBuilder} field.
     *
     * {@link Attribute.Builder} class is not thread-safe for concurrent modification.
     */
    private final Object lock = new Object();

    private final String name;

    public Descriptor(String name, Attribute.Builder<?> attributeBuilder) {
      this.attributeBuilder = attributeBuilder;
      this.name = name;
    }

    public boolean hasDefault() {
      synchronized (lock) {
        return attributeBuilder.isValueSet();
      }
    }

    public AttributeValueSource getValueSource() {
      synchronized (lock) {
        return attributeBuilder.getValueSource();
      }
    }

    public Attribute build(String name) {
      synchronized (lock) {
        return attributeBuilder.build(name);
      }
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("<attr." + name + ">");
    }
  }

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkAttr.class);
  }
}
