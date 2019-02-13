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

package com.google.devtools.build.lib.packages;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.analysis.skylark.BazelStarlarkContext;
import com.google.devtools.build.lib.analysis.skylark.SymbolGenerator;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Globber.BadGlobException;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.syntax.SkylarkUtils;
import com.google.devtools.build.lib.syntax.SkylarkUtils.Phase;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * The package factory is responsible for constructing Package instances
 * from a BUILD file's abstract syntax tree (AST).
 *
 * <p>A PackageFactory is a heavy-weight object; create them sparingly.
 * Typically only one is needed per client application.
 */
public final class PackageFactory {
  /**
   * An argument to the {@code package()} function.
   */
  public abstract static class PackageArgument<T> {
    private final String name;
    private final Type<T> type;

    protected PackageArgument(String name, Type<T> type) {
      this.name = name;
      this.type = type;
    }

    public String getName() {
      return name;
    }

    private void convertAndProcess(
        Package.Builder pkgBuilder, Location location, Object value)
        throws EvalException {
      T typedValue = type.convert(value, "'package' argument", pkgBuilder.getBuildFileLabel());
      process(pkgBuilder, location, typedValue);
    }

    /**
     * Process an argument.
     *
     * @param pkgBuilder the package builder to be mutated
     * @param location the location of the {@code package} function for error reporting
     * @param value the value of the argument. Typically passed to {@link Type#convert}
     */
    protected abstract void process(
        Package.Builder pkgBuilder, Location location, T value)
        throws EvalException;
  }

  /**
   * An extension to the global namespace of the BUILD language.
   */
  // TODO(bazel-team): this is largely unrelated to syntax.Environment.Extension,
  // and should probably be renamed PackageFactory.RuntimeExtension, since really,
  // we're extending the Runtime with more classes.
  public interface EnvironmentExtension {
    /**
     * Update the global environment with the identifiers this extension contributes.
     */
    void update(Environment environment);

    /**
     * Update the global environment of WORKSPACE files.
     */
    void updateWorkspace(Environment environment);

    /**
     * Returns the extra functions needed to be added to the Skylark native module.
     */
    ImmutableList<BaseFunction> nativeModuleFunctions();

    /**
     * Returns the extra arguments to the {@code package()} statement.
     */
    Iterable<PackageArgument<?>> getPackageArguments();
  }

  private static class DefaultVisibility extends PackageArgument<List<Label>> {
    private DefaultVisibility() {
      super("default_visibility", BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<Label> value) throws EvalException{
      try {
        pkgBuilder.setDefaultVisibility(getVisibility(pkgBuilder.getBuildFileLabel(), value));
      } catch (EvalException e) {
        throw new EvalException(location, e.getMessage());
      }
    }
  }

  private static class DefaultTestOnly extends PackageArgument<Boolean> {
    private DefaultTestOnly() {
      super("default_testonly", Type.BOOLEAN);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        Boolean value) {
      pkgBuilder.setDefaultTestonly(value);
    }
  }

  private static class DefaultDeprecation extends PackageArgument<String> {
    private DefaultDeprecation() {
      super("default_deprecation", Type.STRING);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        String value) {
      pkgBuilder.setDefaultDeprecation(value);
    }
  }

  private static class Features extends PackageArgument<List<String>> {
    private Features() {
      super("features", Type.STRING_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<String> value) {
      pkgBuilder.addFeatures(value);
    }
  }

  private static class DefaultLicenses extends PackageArgument<License> {
    private DefaultLicenses() {
      super("licenses", BuildType.LICENSE);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        License value) {
      pkgBuilder.setDefaultLicense(value);
    }
  }

  private static class DefaultDistribs extends PackageArgument<Set<DistributionType>> {
    private DefaultDistribs() {
      super("distribs", BuildType.DISTRIBUTIONS);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        Set<DistributionType> value) {
      pkgBuilder.setDefaultDistribs(value);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for
   * {@link RuleClass#COMPATIBLE_ENVIRONMENT_ATTR} when not explicitly specified.
   */
  private static class DefaultCompatibleWith extends PackageArgument<List<Label>> {
    private DefaultCompatibleWith() {
      super(Package.DEFAULT_COMPATIBLE_WITH_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultCompatibleWith(value, Package.DEFAULT_COMPATIBLE_WITH_ATTRIBUTE,
          location);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for
   * {@link RuleClass#RESTRICTED_ENVIRONMENT_ATTR} when not explicitly specified.
   */
  private static class DefaultRestrictedTo extends PackageArgument<List<Label>> {
    private DefaultRestrictedTo() {
      super(Package.DEFAULT_RESTRICTED_TO_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultRestrictedTo(value, Package.DEFAULT_RESTRICTED_TO_ATTRIBUTE, location);
    }
  }

  public static final String PKG_CONTEXT = "$pkg_context";

  /** {@link Globber} that uses the legacy GlobCache. */
  public static class LegacyGlobber implements Globber {
    private final GlobCache globCache;
    private final boolean sort;

    private LegacyGlobber(GlobCache globCache, boolean sort) {
      this.globCache = globCache;
      this.sort = sort;
    }

    private static class Token extends Globber.Token {
      public final List<String> includes;
      public final List<String> excludes;
      public final boolean excludeDirs;

      public Token(List<String> includes, List<String> excludes, boolean excludeDirs) {
        this.includes = includes;
        this.excludes = excludes;
        this.excludeDirs = excludeDirs;
      }
    }

    @Override
    public Token runAsync(List<String> includes, List<String> excludes, boolean excludeDirs)
        throws BadGlobException {
      for (String pattern : Iterables.concat(includes, excludes)) {
        @SuppressWarnings("unused")
        Future<?> possiblyIgnoredError = globCache.getGlobUnsortedAsync(pattern, excludeDirs);
      }
      return new Token(includes, excludes, excludeDirs);
    }

    @Override
    public List<String> fetch(Globber.Token token) throws IOException, InterruptedException {
      List<String> result;
      Token legacyToken = (Token) token;
      try {
        result = globCache.globUnsorted(legacyToken.includes, legacyToken.excludes,
            legacyToken.excludeDirs);
      } catch (BadGlobException e) {
        throw new IllegalStateException(e);
      }
      if (sort) {
        Collections.sort(result);
      }
      return result;
    }

    @Override
    public void onInterrupt() {
      globCache.cancelBackgroundTasks();
    }

    @Override
    public void onCompletion() {
      globCache.finishBackgroundTasks();
    }
  }

  private static final Logger logger = Logger.getLogger(PackageFactory.class.getName());

  private final RuleFactory ruleFactory;
  private final ImmutableMap<String, BuiltinRuleFunction> ruleFunctions;
  private final RuleClassProvider ruleClassProvider;

  private AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls;

  private final ThreadPoolExecutor threadPool;

  private int maxDirectoriesToEagerlyVisitInGlobbing;

  private final ImmutableList<EnvironmentExtension> environmentExtensions;
  private final ImmutableMap<String, PackageArgument<?>> packageArguments;

  private final Package.Builder.Helper packageBuilderHelper;

  /** Builder for {@link PackageFactory} instances. Intended to only be used by unit tests. */
  @VisibleForTesting
  public abstract static class BuilderForTesting {
    protected final String version = "test";
    protected Iterable<EnvironmentExtension> environmentExtensions = ImmutableList.of();
    protected Function<RuleClass, AttributeContainer> attributeContainerFactory =
        AttributeContainer::new;
    protected boolean doChecksForTesting = true;

    public BuilderForTesting setEnvironmentExtensions(
        Iterable<EnvironmentExtension> environmentExtensions) {
      this.environmentExtensions = environmentExtensions;
      return this;
    }

    public BuilderForTesting disableChecks() {
      this.doChecksForTesting = false;
      return this;
    }

    public abstract PackageFactory build(RuleClassProvider ruleClassProvider, FileSystem fs);
  }

  @VisibleForTesting
  public Package.Builder.Helper getPackageBuilderHelperForTesting() {
    return packageBuilderHelper;
  }

  /**
   * Constructs a {@code PackageFactory} instance with a specific glob path translator
   * and rule factory.
   *
   * <p>Only intended to be called by BlazeRuntime or {@link BuilderForTesting#build}.
   *
   * <p>Do not call this constructor directly in tests; please use
   * TestConstants#PACKAGE_FACTORY_BUILDER_FACTORY_FOR_TESTING instead.
   */
  public PackageFactory(
      RuleClassProvider ruleClassProvider,
      Function<RuleClass, AttributeContainer> attributeContainerFactory,
      Iterable<EnvironmentExtension> environmentExtensions,
      String version,
      Package.Builder.Helper packageBuilderHelper) {
    this.ruleFactory = new RuleFactory(ruleClassProvider, attributeContainerFactory);
    this.ruleFunctions = buildRuleFunctions(ruleFactory);
    this.ruleClassProvider = ruleClassProvider;
    threadPool =
        new ThreadPoolExecutor(
            100,
            Integer.MAX_VALUE,
            15L,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<Runnable>(),
            new ThreadFactoryBuilder().setNameFormat("Legacy globber %d").setDaemon(true).build());
    // Do not consume threads when not in use.
    threadPool.allowCoreThreadTimeOut(true);
    this.environmentExtensions = ImmutableList.copyOf(environmentExtensions);
    this.packageArguments = createPackageArguments();
    this.nativeModule = newNativeModule();
    this.workspaceNativeModule = WorkspaceFactory.newNativeModule(ruleClassProvider, version);
    this.packageBuilderHelper = packageBuilderHelper;
  }

 /**
   * Sets the syscalls cache used in globbing.
   */
  public void setSyscalls(AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls) {
    this.syscalls = Preconditions.checkNotNull(syscalls);
  }

  /**
   * Sets the max number of threads to use for globbing.
   */
  public void setGlobbingThreads(int globbingThreads) {
    threadPool.setCorePoolSize(globbingThreads);
  }

  /**
   * Sets the number of directories to eagerly traverse on the first glob for a given package, in
   * order to warm the filesystem. -1 means do no eager traversal. See {@code
   * PackageCacheOptions#maxDirectoriesToEagerlyVisitInGlobbing}.
   */
  public void setMaxDirectoriesToEagerlyVisitInGlobbing(
      int maxDirectoriesToEagerlyVisitInGlobbing) {
    this.maxDirectoriesToEagerlyVisitInGlobbing = maxDirectoriesToEagerlyVisitInGlobbing;
  }

  /**
   * Returns the immutable, unordered set of names of all the known rule
   * classes.
   */
  public Set<String> getRuleClassNames() {
    return ruleFactory.getRuleClassNames();
  }

  /**
   * Returns the {@link RuleClass} for the specified rule class name.
   */
  public RuleClass getRuleClass(String ruleClassName) {
    return ruleFactory.getRuleClass(ruleClassName);
  }

  /**
   * Returns the {@link RuleClassProvider} of this {@link PackageFactory}.
   */
  public RuleClassProvider getRuleClassProvider() {
    return ruleClassProvider;
  }

  public ImmutableList<EnvironmentExtension> getEnvironmentExtensions() {
    return environmentExtensions;
  }

  /**
   * Creates the list of arguments for the 'package' function.
   */
  private ImmutableMap<String, PackageArgument<?>> createPackageArguments() {
    ImmutableList.Builder<PackageArgument<?>> arguments =
        ImmutableList.<PackageArgument<?>>builder()
            .add(new DefaultDeprecation())
            .add(new DefaultDistribs())
            .add(new DefaultLicenses())
            .add(new DefaultTestOnly())
            .add(new DefaultVisibility())
            .add(new Features())
            .add(new DefaultCompatibleWith())
            .add(new DefaultRestrictedTo());

    for (EnvironmentExtension extension : environmentExtensions) {
      arguments.addAll(extension.getPackageArguments());
    }

    ImmutableMap.Builder<String, PackageArgument<?>> packageArguments = ImmutableMap.builder();
    for (PackageArgument<?> argument : arguments.build()) {
      packageArguments.put(argument.getName(), argument);
    }
    return packageArguments.build();
  }

  /**
   * ************************************************************************** Environment function
   * factories.
   */

  /**
   * Returns a function-value implementing "glob" in the specified package context.
   *
   * @param async if true, start globs in the background but don't block on their completion. Only
   *     use this for heuristic preloading.
   */
  @SkylarkSignature(
    name = "glob",
    objectType = Object.class,
    returnType = SkylarkList.class,
    doc = "Returns a list of files that match glob search pattern.",
    parameters = {
      @Param(
        name = "include",
        type = SkylarkList.class,
        generic1 = String.class,
        doc = "a list of strings specifying patterns of files to include."
      ),
      @Param(
        name = "exclude",
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        positional = false,
        named = true,
        doc = "a list of strings specifying patterns of files to exclude."
      ),
      // TODO(bazel-team): migrate all existing code to use boolean instead?
      @Param(
        name = "exclude_directories",
        type = Integer.class,
        defaultValue = "1",
        positional = false,
        named = true,
        doc = "a integer that if non-zero indicates directories should not be matched."
      )
    },
    documented = false,
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction.Factory newGlobFunction =
      new BuiltinFunction.Factory("glob") {
        public BuiltinFunction create(final PackageContext originalContext) {
          return new BuiltinFunction("glob", this) {
            public SkylarkList invoke(
                SkylarkList include,
                SkylarkList exclude,
                Integer excludeDirectories,
                FuncallExpression ast,
                Environment env)
                throws EvalException, ConversionException, InterruptedException {
              return callGlob(originalContext, include, exclude, excludeDirectories != 0, ast, env);
            }
          };
        }
      };

  static SkylarkList<Object> callGlob(
      @Nullable PackageContext originalContext,
      Object include,
      Object exclude,
      boolean excludeDirs,
      FuncallExpression ast,
      Environment env)
      throws EvalException, ConversionException, InterruptedException {
    // Skylark build extensions need to get the PackageContext from the Environment;
    // async glob functions cannot do the same because the Environment is not thread safe.
    PackageContext context;
    if (originalContext == null) {
      context = getContext(env, ast.getLocation());
    } else {
      context = originalContext;
    }

    List<String> includes = Type.STRING_LIST.convert(include, "'glob' argument");
    List<String> excludes = Type.STRING_LIST.convert(exclude, "'glob' argument");

    List<String> matches;
    try {
      Globber.Token globToken = context.globber.runAsync(includes, excludes, excludeDirs);
      matches = context.globber.fetch(globToken);
    } catch (IOException e) {
      String errorMessage = String.format(
          "error globbing [%s]%s: %s",
          Joiner.on(", ").join(includes),
          excludes.isEmpty() ? "" : " - [" + Joiner.on(", ").join(excludes) + "]",
          e.getMessage());
      context.eventHandler.handle(Event.error(ast.getLocation(), errorMessage));
      context.pkgBuilder.setIOExceptionAndMessage(e, errorMessage);
      matches = ImmutableList.of();
    } catch (BadGlobException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }

    return MutableList.copyOf(env, matches);
  }

  /**
   * Returns a function value implementing "environment_group" in the specified package context.
   * Syntax is as follows:
   *
   * <pre>{@code
   *   environment_group(
   *       name = "sample_group",
   *       environments = [":env1", ":env2", ...],
   *       defaults = [":env1", ...]
   *   )
   * }</pre>
   *
   * <p>Where ":env1", "env2", ... are all environment rules declared in the same package. All
   * parameters are mandatory.
   */
  @SkylarkSignature(name = "environment_group", returnType = Runtime.NoneType.class,
      doc = "Defines a set of related environments that can be tagged onto rules to prevent"
      + "incompatible rules from depending on each other.",
      parameters = {
        @Param(name = "name", type = String.class, positional = false, named = true,
            doc = "The name of the rule."),
        // Both parameter below are lists of label designators
        @Param(name = "environments", type = SkylarkList.class, generic1 = Object.class,
            positional = false, named = true,
            doc = "A list of Labels for the environments to be grouped, from the same package."),
        @Param(name = "defaults", type = SkylarkList.class, generic1 = Object.class,
            positional = false, named = true,
            doc = "A list of Labels.")}, // TODO(bazel-team): document what that is
      documented = false, useLocation = true)
  private static final BuiltinFunction.Factory newEnvironmentGroupFunction =
      new BuiltinFunction.Factory("environment_group") {
        public BuiltinFunction create(final PackageContext context) {
          return new BuiltinFunction("environment_group", this) {
            public Runtime.NoneType invoke(
                String name, SkylarkList environmentsList, SkylarkList defaultsList,
                Location loc) throws EvalException, ConversionException {
              List<Label> environments = BuildType.LABEL_LIST.convert(environmentsList,
                  "'environment_group argument'", context.pkgBuilder.getBuildFileLabel());
              List<Label> defaults = BuildType.LABEL_LIST.convert(defaultsList,
                  "'environment_group argument'", context.pkgBuilder.getBuildFileLabel());

              if (environments.isEmpty()) {
                throw new EvalException(location,
                    "environment group " + name + " must contain at least one environment");
              }
              try {
                context.pkgBuilder.addEnvironmentGroup(
                    name, environments, defaults, context.eventHandler, loc);
                return Runtime.NONE;
              } catch (LabelSyntaxException e) {
                throw new EvalException(loc,
                    "environment group has invalid name: " + name + ": " + e.getMessage());
              } catch (Package.NameConflictException e) {
                throw new EvalException(loc, e.getMessage());
              }
            }
          };
        }
      };

  /** Returns a function-value implementing "exports_files" in the specified package context. */
  @SkylarkSignature(
    name = "exports_files",
    returnType = Runtime.NoneType.class,
    doc = "Declare a set of files as exported",
    parameters = {
      @Param(
        name = "srcs",
        type = SkylarkList.class,
        generic1 = String.class,
        doc = "A list of strings, the names of the files to export."
      ),
      // TODO(blaze-team): make it possible to express a list of label designators,
      // i.e. a java List or Skylark list of Label or String.
      @Param(
        name = "visibility",
        type = SkylarkList.class,
        noneable = true,
        defaultValue = "None",
        doc =
            "A list of Labels specifying the visibility of the exported files "
                + "(defaults to public)."
      ),
      @Param(
        name = "licenses",
        type = SkylarkList.class,
        generic1 = String.class,
        noneable = true,
        defaultValue = "None",
        doc = "A list of strings specifying the licenses used in the exported code."
      )
    },
    documented = false,
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction.Factory newExportsFilesFunction =
      new BuiltinFunction.Factory("exports_files") {
        public BuiltinFunction create() {
          return new BuiltinFunction("exports_files", this) {
            public Runtime.NoneType invoke(
                SkylarkList srcs,
                Object visibility,
                Object licenses,
                FuncallExpression ast,
                Environment env)
                throws EvalException, ConversionException {
              return callExportsFiles(srcs, visibility, licenses, ast, env);
            }
          };
        }
      };

  static Runtime.NoneType callExportsFiles(Object srcs, Object visibilityO, Object licensesO,
      FuncallExpression ast, Environment env) throws EvalException, ConversionException {
    Package.Builder pkgBuilder = getContext(env, ast.getLocation()).pkgBuilder;
    List<String> files = Type.STRING_LIST.convert(srcs, "'exports_files' operand");

    RuleVisibility visibility;
    try {
      visibility = EvalUtils.isNullOrNone(visibilityO)
          ? ConstantRuleVisibility.PUBLIC
          : getVisibility(pkgBuilder.getBuildFileLabel(), BuildType.LABEL_LIST.convert(
              visibilityO,
              "'exports_files' operand",
              pkgBuilder.getBuildFileLabel()));
    } catch (EvalException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
    // TODO(bazel-team): is licenses plural or singular?
    License license = BuildType.LICENSE.convertOptional(licensesO, "'exports_files' operand");

    for (String file : files) {
      String errorMessage = LabelValidator.validateTargetName(file);
      if (errorMessage != null) {
        throw new EvalException(ast.getLocation(), errorMessage);
      }
      try {
        InputFile inputFile = pkgBuilder.createInputFile(file, ast.getLocation());
        if (inputFile.isVisibilitySpecified()
            && inputFile.getVisibility() != visibility) {
          throw new EvalException(ast.getLocation(),
              String.format("visibility for exported file '%s' declared twice",
                  inputFile.getName()));
        }
        if (license != null && inputFile.isLicenseSpecified()) {
          throw new EvalException(ast.getLocation(),
              String.format("licenses for exported file '%s' declared twice",
                  inputFile.getName()));
        }
        if (env.getSemantics().checkThirdPartyTargetsHaveLicenses()
            && license == null
            && !pkgBuilder.getDefaultLicense().isSpecified()
            && RuleClass.isThirdPartyPackage(pkgBuilder.getPackageIdentifier())) {
          throw new EvalException(ast.getLocation(),
              "third-party file '" + inputFile.getName() + "' lacks a license declaration "
              + "with one of the following types: notice, reciprocal, permissive, "
              + "restricted, unencumbered, by_exception_only");
        }

        pkgBuilder.setVisibilityAndLicense(inputFile, visibility, license);
      } catch (Package.Builder.GeneratedLabelConflict e) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
    }
    return Runtime.NONE;
  }

  /**
   * Returns a function-value implementing "licenses" in the specified package
   * context.
   * TODO(bazel-team): Remove in favor of package.licenses.
   */
  @SkylarkSignature(name = "licenses", returnType = Runtime.NoneType.class,
      doc = "Declare the license(s) for the code in the current package.",
      parameters = {
        @Param(name = "license_strings", type = SkylarkList.class, generic1 = String.class,
            doc = "A list of strings, the names of the licenses used.")},
      documented = false, useLocation = true)
  private static final BuiltinFunction.Factory newLicensesFunction =
      new BuiltinFunction.Factory("licenses") {
        public BuiltinFunction create(final PackageContext context) {
          return new BuiltinFunction("licenses", this) {
            public Runtime.NoneType invoke(SkylarkList licensesList, Location loc) {
              try {
                License license = BuildType.LICENSE.convert(licensesList, "'licenses' operand");
                context.pkgBuilder.setDefaultLicense(license);
              } catch (ConversionException e) {
                context.eventHandler.handle(Event.error(loc, e.getMessage()));
                context.pkgBuilder.setContainsErrors();
              }
              return Runtime.NONE;
            }
          };
        }
      };

  /**
   * Returns a function-value implementing "distribs" in the specified package
   * context.
   */
  // TODO(bazel-team): Remove in favor of package.distribs.
  // TODO(bazel-team): Remove all these new*Function-s and/or have static functions
  // that consult the context dynamically via getContext(env, loc) since we have that,
  // and share the functions with the native package... which requires unifying the List types.
  @SkylarkSignature(name = "distribs", returnType = Runtime.NoneType.class,
      doc = "Declare the distribution(s) for the code in the current package.",
      parameters = {
        @Param(name = "distribution_strings", type = Object.class,
            doc = "The distributions.")},
      documented = false, useLocation = true)
  private static final BuiltinFunction.Factory newDistribsFunction =
      new BuiltinFunction.Factory("distribs") {
        public BuiltinFunction create(final PackageContext context) {
          return new BuiltinFunction("distribs", this) {
            public Runtime.NoneType invoke(Object object, Location loc) {
              try {
                Set<DistributionType> distribs = BuildType.DISTRIBUTIONS.convert(object,
                    "'distribs' operand");
                context.pkgBuilder.setDefaultDistribs(distribs);
              } catch (ConversionException e) {
                context.eventHandler.handle(Event.error(loc, e.getMessage()));
                context.pkgBuilder.setContainsErrors();
              }
              return Runtime.NONE;
            }
          };
        }
      };

  @SkylarkSignature(
    name = "package_group",
    returnType = Runtime.NoneType.class,
    doc = "Declare a set of files as exported.",
    parameters = {
      @Param(
        name = "name",
        type = String.class,
        named = true,
        positional = false,
        doc = "The name of the rule."
      ),
      @Param(
        name = "packages",
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = "A list of Strings specifying the packages grouped."
      ),
      // java list or list of label designators: Label or String
      @Param(
        name = "includes",
        type = SkylarkList.class,
        generic1 = Object.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = "A list of Label specifiers for the files to include."
      )
    },
    documented = false,
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction.Factory newPackageGroupFunction =
      new BuiltinFunction.Factory("package_group") {
        public BuiltinFunction create() {
          return new BuiltinFunction("package_group", this) {
            public Runtime.NoneType invoke(
                String name,
                SkylarkList packages,
                SkylarkList includes,
                FuncallExpression ast,
                Environment env)
                throws EvalException, ConversionException {
              return callPackageFunction(name, packages, includes, ast, env);
            }
          };
        }
      };

  @Nullable
  static SkylarkDict<String, Object> callGetRuleFunction(
      String name, FuncallExpression ast, Environment env)
      throws EvalException, ConversionException {
    PackageContext context = getContext(env, ast.getLocation());
    Target target = context.pkgBuilder.getTarget(name);

    return targetDict(target, ast.getLocation(), env);
  }

  @Nullable
  private static SkylarkDict<String, Object> targetDict(
      Target target, Location loc, Environment env)
      throws NotRepresentableException, EvalException {
    if (target == null || !(target instanceof Rule)) {
      return null;
    }
    SkylarkDict<String, Object> values = SkylarkDict.<String, Object>of(env);

    Rule rule = (Rule) target;
    AttributeContainer cont = rule.getAttributeContainer();
    for (Attribute attr : rule.getAttributes()) {
      if (!Character.isAlphabetic(attr.getName().charAt(0))) {
        continue;
      }

      if (attr.getName().equals("distribs")) {
        // attribute distribs: cannot represent type class java.util.Collections$SingletonSet
        // in Skylark: [INTERNAL].
        continue;
      }

      try {
        Object val = skylarkifyValue(cont.getAttr(attr.getName()), target.getPackage());
        if (val == null) {
          continue;
        }
        values.put(attr.getName(), val, loc, env);
      } catch (NotRepresentableException e) {
        throw new NotRepresentableException(
            String.format(
                "target %s, attribute %s: %s", target.getName(), attr.getName(), e.getMessage()));
      }
    }

    values.put("name", rule.getName(), loc, env);
    values.put("kind", rule.getRuleClass(), loc, env);
    return values;
  }

  static class NotRepresentableException extends EvalException {
    NotRepresentableException(String msg) {
      super(null, msg);
    }
  };

  /**
   * Converts back to type that will work in BUILD and skylark,
   * such as string instead of label, SkylarkList instead of List,
   * Returns null if we don't want to export the value.
   *
   * <p>All of the types returned are immutable. If we want, we can change this to
   * immutable in the future, but this is the safe choice for now.
   */
  @Nullable
  private static Object skylarkifyValue(Object val, Package pkg) throws NotRepresentableException {
    // TODO(bazel-team): the location of this function is ad-hoc. Arguably, the conversion
    // from Java native types to Skylark types should be part of the Type class hierarchy,
    if (val == null) {
      return null;
    }
    if (val instanceof Boolean) {
      return val;
    }
    if (val instanceof Integer) {
      return val;
    }
    if (val instanceof String) {
      return val;
    }

    if (val instanceof TriState) {
      switch ((TriState) val) {
        case AUTO:
          return Integer.valueOf(-1);
        case YES:
          return Integer.valueOf(1);
        case NO:
          return Integer.valueOf(0);
      }
    }

    if (val instanceof Label) {
      Label l = (Label) val;
      if (l.getPackageName().equals(pkg.getName())) {
        return ":" + l.getName();
      }
      return l.getCanonicalForm();
    }

    if (val instanceof List) {
      List<Object> l = new ArrayList<>();
      for (Object o : (List) val) {
        Object elt = skylarkifyValue(o, pkg);
        if (elt == null) {
          continue;
        }

        l.add(elt);
      }

      return SkylarkList.Tuple.copyOf(l);
    }
    if (val instanceof Map) {
      Map<Object, Object> m = new TreeMap<>();
      for (Map.Entry<?, ?> e : ((Map<?, ?>) val).entrySet()) {
        Object key = skylarkifyValue(e.getKey(), pkg);
        Object mapVal = skylarkifyValue(e.getValue(), pkg);

        if (key == null || mapVal == null) {
          continue;
        }

        m.put(key, mapVal);
      }
      return m;
    }
    if (val.getClass().isAnonymousClass()) {
      // Computed defaults. They will be represented as
      // "deprecation": com.google.devtools.build.lib.analysis.BaseRuleClasses$2@6960884a,
      // Filter them until we invent something more clever.
      return null;
    }

    if (val instanceof License) {
      // License is deprecated as a Starlark type, so omit this type from Starlark values
      // to avoid exposing these objects, even though they are technically SkylarkValue.
      return null;
    }

    if (val instanceof SkylarkValue) {
      return val;
    }

    if (val instanceof BuildType.SelectorList) {
      // This is terrible:
      //  1) this value is opaque, and not a BUILD value, so it cannot be used in rule arguments
      //  2) its representation has a pointer address, so it breaks hermeticity.
      //
      // Even though this is clearly imperfect, we return this value because otherwise
      // native.rules() fails if there is any rule using a select() in the BUILD file.
      //
      // To remedy this, we should return a syntax.SelectorList. To do so, we have to
      // 1) recurse into the Selector contents of SelectorList, so those values are skylarkified too
      // 2) get the right Class<?> value. We could probably get at that by looking at
      //    ((SelectorList)val).getSelectors().first().getEntries().first().getClass().

      return val;
    }

    // We are explicit about types we don't understand so we minimize changes to existing callers
    // if we add more types that we can represent.
    throw new NotRepresentableException(
        String.format("cannot represent %s (%s) in Starlark", val, val.getClass()));
  }


  static SkylarkDict<String, SkylarkDict<String, Object>> callGetRulesFunction(
      FuncallExpression ast, Environment env)
      throws EvalException {
    PackageContext context = getContext(env, ast.getLocation());
    Collection<Target> targets = context.pkgBuilder.getTargets();
    Location loc = ast.getLocation();

    SkylarkDict<String, SkylarkDict<String, Object>> rules = SkylarkDict.of(env);
    for (Target t : targets) {
      if (t instanceof Rule) {
        SkylarkDict<String, Object> m = targetDict(t, loc, env);
        Preconditions.checkNotNull(m);
        rules.put(t.getName(), m, loc, env);
      }
    }

    return rules;
  }

  static Runtime.NoneType callPackageFunction(String name, Object packagesO, Object includesO,
      FuncallExpression ast, Environment env) throws EvalException, ConversionException {
    PackageContext context = getContext(env, ast.getLocation());

    List<String> packages = Type.STRING_LIST.convert(
        packagesO, "'package_group.packages argument'");
    List<Label> includes = BuildType.LABEL_LIST.convert(includesO,
        "'package_group.includes argument'", context.pkgBuilder.getBuildFileLabel());

    try {
      context.pkgBuilder.addPackageGroup(name, packages, includes, context.eventHandler,
          ast.getLocation());
      return Runtime.NONE;
    } catch (LabelSyntaxException e) {
      throw new EvalException(ast.getLocation(),
          "package group has invalid name: " + name + ": " + e.getMessage());
    } catch (Package.NameConflictException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  public static RuleVisibility getVisibility(Label ruleLabel, List<Label> original)
      throws EvalException {
    RuleVisibility result;

    result = ConstantRuleVisibility.tryParse(original);
    if (result != null) {
      return result;
    }

    result = PackageGroupsRuleVisibility.tryParse(ruleLabel, original);
    return result;
  }

  /**
   * Returns a function-value implementing "package" in the specified package
   * context.
   */
  private static BaseFunction newPackageFunction(
      final ImmutableMap<String, PackageArgument<?>> packageArguments) {
    // Flatten the map of argument name of PackageArgument specifier in two co-indexed arrays:
    // one for the argument names, to create a FunctionSignature when we create the function,
    // one of the PackageArgument specifiers, over which to iterate at every function invocation
    // at the same time that we iterate over the function arguments.
    final int numArgs = packageArguments.size();
    final String[] argumentNames = new String[numArgs];
    final PackageArgument<?>[] argumentSpecifiers = new PackageArgument<?>[numArgs];
    int i = 0;
    for (Map.Entry<String, PackageArgument<?>> entry : packageArguments.entrySet()) {
      argumentNames[i] = entry.getKey();
      argumentSpecifiers[i++] = entry.getValue();
    }

    return new BaseFunction("package", FunctionSignature.namedOnly(0, argumentNames)) {
      @Override
      public Object call(Object[] arguments, FuncallExpression ast, Environment env)
          throws EvalException {

        Package.Builder pkgBuilder = getContext(env, ast.getLocation()).pkgBuilder;

        // Validate parameter list
        if (pkgBuilder.isPackageFunctionUsed()) {
          throw new EvalException(ast.getLocation(),
              "'package' can only be used once per BUILD file");
        }
        pkgBuilder.setPackageFunctionUsed();

        // Parse params
        boolean foundParameter = false;

        for (int i = 0; i < numArgs; i++) {
          Object value = arguments[i];
          if (value != null) {
            foundParameter = true;
            argumentSpecifiers[i].convertAndProcess(pkgBuilder, ast.getLocation(), value);
          }
        }

        if (!foundParameter) {
          throw new EvalException(ast.getLocation(),
              "at least one argument must be given to the 'package' function");
        }

        return Runtime.NONE;
      }
    };
  }


  /**
   * Get the PackageContext by looking up in the environment.
   */
  public static PackageContext getContext(Environment env, Location location)
      throws EvalException {
    PackageContext value = (PackageContext) env.dynamicLookup(PKG_CONTEXT);
    if (value == null) {
      // if PKG_CONTEXT is missing, we're not called from a BUILD file. This happens if someone
      // uses native.some_func() in the wrong place.
      throw new EvalException(location,
          "The native module cannot be accessed from here. "
          + "Wrap the function in a macro and call it from a BUILD file");
    }
    return value;
  }

  private static ImmutableMap<String, BuiltinRuleFunction> buildRuleFunctions(
      RuleFactory ruleFactory) {
    ImmutableMap.Builder<String, BuiltinRuleFunction> result = ImmutableMap.builder();
    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      result.put(ruleClass, new BuiltinRuleFunction(ruleClass, ruleFactory));
    }
    return result.build();
  }

  /** {@link BuiltinFunction} adapter for creating {@link Rule}s for native {@link RuleClass}es. */
  private static class BuiltinRuleFunction extends BuiltinFunction implements RuleFunction {
    private final String ruleClassName;
    private final RuleFactory ruleFactory;
    private final RuleClass ruleClass;

    BuiltinRuleFunction(String ruleClassName, RuleFactory ruleFactory) {
      super(ruleClassName, FunctionSignature.KWARGS, BuiltinFunction.USE_AST_ENV, /*isRule=*/ true);
      this.ruleClassName = ruleClassName;
      this.ruleFactory = Preconditions.checkNotNull(ruleFactory, "ruleFactory was null");
      this.ruleClass = Preconditions.checkNotNull(
          ruleFactory.getRuleClass(ruleClassName),
          "No such rule class: %s",
          ruleClassName);
    }

    @SuppressWarnings({"unchecked", "unused"})
    public Runtime.NoneType invoke(
        Map<String, Object> kwargs, FuncallExpression ast, Environment env)
        throws EvalException, InterruptedException {
      SkylarkUtils.checkLoadingOrWorkspacePhase(env, ruleClassName, ast.getLocation());
      try {
        addRule(getContext(env, ast.getLocation()), kwargs, ast, env);
      } catch (RuleFactory.InvalidRuleException | Package.NameConflictException e) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
      return Runtime.NONE;
    }

    private void addRule(
        PackageContext context,
        Map<String, Object> kwargs,
        FuncallExpression ast,
        Environment env)
            throws RuleFactory.InvalidRuleException, Package.NameConflictException,
                InterruptedException {
      BuildLangTypedAttributeValuesMap attributeValues =
          new BuildLangTypedAttributeValuesMap(kwargs);
      AttributeContainer attributeContainer = ruleFactory.getAttributeContainer(ruleClass);
      RuleFactory.createAndAddRule(
          context, ruleClass, attributeValues, ast, env, attributeContainer);
    }

    @Override
    public RuleClass getRuleClass() {
      return ruleClass;
    }
  }

  /**
   * Loads, scans parses and evaluates the build file at "buildFile", and creates and returns a
   * Package builder instance capable of building a package identified by "packageId".
   *
   * <p>This method returns a builder to allow the caller to do additional work, if necessary.
   *
   * <p>This method assumes "packageId" is a valid package name according to the {@link
   * LabelValidator#validatePackageName} heuristic.
   *
   * <p>See {@link #evaluateBuildFile} for information on AST retention.
   *
   * <p>Executes {@code globber.onCompletion()} on completion and executes {@code
   * globber.onInterrupt()} on an {@link InterruptedException}.
   */
  private Package.Builder createPackage(
      String workspaceName,
      PackageIdentifier packageId,
      RootedPath buildFile,
      ParserInputSource input,
      List<Statement> preludeStatements,
      Map<String, Extension> imports,
      ImmutableList<Label> skylarkFileDependencies,
      RuleVisibility defaultVisibility,
      SkylarkSemantics skylarkSemantics,
      Globber globber)
      throws InterruptedException {
    StoredEventHandler localReporterForParsing = new StoredEventHandler();
    // Run the lexer and parser with a local reporter, so that errors from other threads do not
    // show up below.
    BuildFileAST buildFileAST =
        parseBuildFile(
            packageId,
            input,
            preludeStatements,
            /* repositoryMapping= */ ImmutableMap.of(),
            localReporterForParsing);
    AstParseResult astParseResult =
        new AstParseResult(buildFileAST, localReporterForParsing);
    return createPackageFromAst(
        workspaceName,
        /* repositoryMapping= */ ImmutableMap.of(),
        packageId,
        buildFile,
        astParseResult,
        imports,
        skylarkFileDependencies,
        defaultVisibility,
        skylarkSemantics,
        globber);
  }

  public static BuildFileAST parseBuildFile(
      PackageIdentifier packageId,
      ParserInputSource in,
      List<Statement> preludeStatements,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      ExtendedEventHandler eventHandler) {
    // Logged messages are used as a testability hook tracing the parsing progress
    logger.fine("Starting to parse " + packageId);
    BuildFileAST buildFileAST =
        BuildFileAST.parseBuildFile(in, preludeStatements, repositoryMapping, eventHandler);
    logger.fine("Finished parsing of " + packageId);
    return buildFileAST;
  }

  public Package.Builder createPackageFromAst(
      String workspaceName,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      PackageIdentifier packageId,
      RootedPath buildFile,
      AstParseResult astParseResult,
      Map<String, Extension> imports,
      ImmutableList<Label> skylarkFileDependencies,
      RuleVisibility defaultVisibility,
      SkylarkSemantics skylarkSemantics,
      Globber globber)
      throws InterruptedException {
    try {
      // At this point the package is guaranteed to exist.  It may have parse or
      // evaluation errors, resulting in a diminished number of rules.
      return evaluateBuildFile(
          workspaceName,
          packageId,
          astParseResult.ast,
          buildFile,
          globber,
          astParseResult.allEvents,
          astParseResult.allPosts,
          defaultVisibility,
          skylarkSemantics,
          imports,
          skylarkFileDependencies,
          repositoryMapping);
    } catch (InterruptedException e) {
      globber.onInterrupt();
      throw e;
    } finally {
      globber.onCompletion();
    }
  }

  @VisibleForTesting
  public Package.Builder newExternalPackageBuilder(
      RootedPath workspacePath, String runfilesPrefix) {
    return Package.newExternalPackageBuilder(packageBuilderHelper, workspacePath, runfilesPrefix);
  }

  @VisibleForTesting
  public Package.Builder newPackageBuilder(PackageIdentifier packageId, String runfilesPrefix) {
    return new Package.Builder(packageBuilderHelper, packageId, runfilesPrefix);
  }

  @VisibleForTesting
  public Package createPackageForTesting(
      PackageIdentifier packageId,
      RootedPath buildFile,
      CachingPackageLocator locator,
      ExtendedEventHandler eventHandler)
      throws NoSuchPackageException, InterruptedException {
    Package externalPkg =
        newExternalPackageBuilder(
                RootedPath.toRootedPath(
                    buildFile.getRoot(),
                    buildFile
                        .getRootRelativePath()
                        .getRelative(LabelConstants.WORKSPACE_FILE_NAME)),
                "TESTING")
            .build();
    return createPackageForTesting(
        packageId,
        externalPkg,
        buildFile,
        locator,
        eventHandler,
        SkylarkSemantics.DEFAULT_SEMANTICS);
  }

  /**
   * Same as createPackage, but does the required validation of "packageName" first, throwing a
   * {@link NoSuchPackageException} if the name is invalid.
   */
  @VisibleForTesting
  public Package createPackageForTesting(
      PackageIdentifier packageId,
      Package externalPkg,
      RootedPath buildFile,
      CachingPackageLocator locator,
      ExtendedEventHandler eventHandler,
      SkylarkSemantics semantics)
      throws NoSuchPackageException, InterruptedException {
    String error =
        LabelValidator.validatePackageName(packageId.getPackageFragment().getPathString());
    if (error != null) {
      throw new BuildFileNotFoundException(
          packageId, "illegal package name: '" + packageId + "' (" + error + ")");
    }
    byte[] buildFileBytes = maybeGetBuildFileBytes(buildFile.asPath(), eventHandler);
    if (buildFileBytes == null) {
      throw new BuildFileContainsErrorsException(packageId, "IOException occurred");
    }

    Globber globber =
        createLegacyGlobber(buildFile.asPath().getParentDirectory(), packageId, locator);
    ParserInputSource input =
        ParserInputSource.create(
            FileSystemUtils.convertFromLatin1(buildFileBytes), buildFile.asPath().asFragment());

    Package result =
        createPackage(
                externalPkg.getWorkspaceName(),
                packageId,
                buildFile,
                input,
                /* preludeStatements= */ ImmutableList.<Statement>of(),
                /* imports= */ ImmutableMap.<String, Extension>of(),
                /* skylarkFileDependencies= */ ImmutableList.<Label>of(),
                /* defaultVisibility= */ ConstantRuleVisibility.PUBLIC,
                semantics,
                globber)
            .build();
    for (Postable post : result.getPosts()) {
      eventHandler.post(post);
    }
    Event.replayEventsOn(eventHandler, result.getEvents());
    return result;
  }

  /** Returns a new {@link LegacyGlobber}. */
  public LegacyGlobber createLegacyGlobber(
      Path packageDirectory,
      PackageIdentifier packageId,
      CachingPackageLocator locator) {
    return createLegacyGlobber(
        new GlobCache(
            packageDirectory,
            packageId,
            locator,
            syscalls,
            threadPool,
            maxDirectoriesToEagerlyVisitInGlobbing));
  }

  /** Returns a new {@link LegacyGlobber}. */
  public static LegacyGlobber createLegacyGlobber(GlobCache globCache) {
    return new LegacyGlobber(globCache, /*sort=*/ true);
  }

  /**
   * Returns a new {@link LegacyGlobber}, the same as in {@link #createLegacyGlobber}, except that
   * the implementation of {@link Globber#fetch} intentionally breaks the contract and doesn't
   * return sorted results.
   */
  public LegacyGlobber createLegacyGlobberThatDoesntSort(
      Path packageDirectory,
      PackageIdentifier packageId,
      CachingPackageLocator locator) {
    return new LegacyGlobber(
        new GlobCache(
            packageDirectory,
            packageId,
            locator,
            syscalls,
            threadPool,
            maxDirectoriesToEagerlyVisitInGlobbing),
        /*sort=*/ false);
  }

  @Nullable
  private byte[] maybeGetBuildFileBytes(Path buildFile, ExtendedEventHandler eventHandler) {
    try {
      return FileSystemUtils.readWithKnownFileSize(buildFile, buildFile.getFileSize());
    } catch (IOException e) {
      eventHandler.handle(Event.error(Location.fromFile(buildFile), e.getMessage()));
      return null;
    }
  }

  /**
   * This tuple holds the current package builder, current lexer, etc, for the
   * duration of the evaluation of one BUILD file. (We use a PackageContext
   * object in preference to storing these values in mutable fields of the
   * PackageFactory.)
   *
   * <p>PLEASE NOTE: references to PackageContext objects are held by many
   * BaseFunction closures, but should become unreachable once the Environment is
   * discarded at the end of evaluation.  Please be aware of your memory
   * footprint when making changes here!
   */
  public static class PackageContext {
    final Package.Builder pkgBuilder;
    final Globber globber;
    final ExtendedEventHandler eventHandler;
    private final Function<RuleClass, AttributeContainer> attributeContainerFactory;

    @VisibleForTesting
    public PackageContext(
        Package.Builder pkgBuilder,
        Globber globber,
        ExtendedEventHandler eventHandler,
        Function<RuleClass, AttributeContainer> attributeContainerFactory) {
      this.pkgBuilder = pkgBuilder;
      this.eventHandler = eventHandler;
      this.globber = globber;
      this.attributeContainerFactory = attributeContainerFactory;
    }

    /**
     * Returns the Label of this Package.
     */
    public Label getLabel() {
      return pkgBuilder.getBuildFileLabel();
    }

    /**
     * Sets a Make variable.
     */
    public void setMakeVariable(String name, String value) {
      pkgBuilder.setMakeVariable(name, value);
    }

    /**
     * Returns the builder of this Package.
     */
    public Package.Builder getBuilder() {
      return pkgBuilder;
    }

    public Function<RuleClass, AttributeContainer> getAttributeContainerFactory() {
      return attributeContainerFactory;
    }
  }

  private final ClassObject nativeModule;
  private final ClassObject workspaceNativeModule;

  /** @return the Skylark struct to bind to "native" */
  public ClassObject getNativeModule(boolean workspace) {
    return workspace ? workspaceNativeModule : nativeModule;
  }

  /**
   * Returns a native module with the functions created using the {@link RuleClassProvider}
   * of this {@link PackageFactory}.
   */
  private ClassObject newNativeModule() {
    ImmutableMap.Builder<String, Object> builder = new ImmutableMap.Builder<>();
    SkylarkNativeModule nativeModuleInstance = new SkylarkNativeModule();
    for (String nativeFunction : FuncallExpression.getMethodNames(SkylarkNativeModule.class)) {
      builder.put(nativeFunction,
          FuncallExpression.getBuiltinCallable(nativeModuleInstance, nativeFunction));
    }
    builder.putAll(ruleFunctions);
    builder.put("package", newPackageFunction(packageArguments));
    for (EnvironmentExtension extension : environmentExtensions) {
      for (BaseFunction function : extension.nativeModuleFunctions()) {
        builder.put(function.getName(), function);
      }
    }
    return StructProvider.STRUCT.create(builder.build(), "no native function or rule '%s'");
  }

  private void buildPkgEnv(Environment pkgEnv, PackageContext context) {
    // TODO(bazel-team): remove the naked functions that are redundant with the nativeModule,
    // or if not possible, at least make them straight copies from the native module variant.
    // or better, use a common Environment.Frame for these common bindings
    // (that shares a backing ImmutableMap for the bindings?)
    Object packageNameFunction;
    Object repositoryNameFunction;
    try {
      packageNameFunction = nativeModule.getValue("package_name");
      repositoryNameFunction = nativeModule.getValue("repository_name");
    } catch (EvalException exception) {
      // This should not occur, as nativeModule.getValue should never throw an exception.
      throw new IllegalStateException(
          "error getting package_name or repository_name functions from the native module",
          exception);
    }
    pkgEnv
        .setup("native", nativeModule)
        .setup("distribs", newDistribsFunction.apply(context))
        .setup("glob", newGlobFunction.apply(context))
        .setup("licenses", newLicensesFunction.apply(context))
        .setup("exports_files", newExportsFilesFunction.apply())
        .setup("package_group", newPackageGroupFunction.apply())
        .setup("package", newPackageFunction(packageArguments))
        .setup("package_name", packageNameFunction)
        .setup("repository_name", repositoryNameFunction)
        .setup("environment_group", newEnvironmentGroupFunction.apply(context));

    for (Map.Entry<String, BuiltinRuleFunction> entry : ruleFunctions.entrySet()) {
      pkgEnv.setup(entry.getKey(), entry.getValue());
    }

    for (EnvironmentExtension extension : environmentExtensions) {
      extension.update(pkgEnv);
    }

    pkgEnv.setupDynamic(PKG_CONTEXT, context);
  }

  /**
   * Called by a caller of {@link #createPackageFromAst} after this caller has fully
   * loaded the package.
   */
  public void afterDoneLoadingPackage(
      Package pkg, SkylarkSemantics skylarkSemantics, long loadTimeNanos) {
    packageBuilderHelper.onLoadingComplete(pkg, skylarkSemantics, loadTimeNanos);
  }

  /**
   * Constructs a Package instance, evaluates the BUILD-file AST inside the build environment, and
   * populates the package with Rule instances as it goes. As with most programming languages,
   * evaluation stops when an exception is encountered: no further rules after the point of failure
   * will be constructed. We assume that rules constructed before the point of failure are valid;
   * this assumption is not entirely correct, since a "vardef" after a rule declaration can affect
   * the behavior of that rule.
   *
   * <p>Rule attribute checking is performed during evaluation. Each attribute must conform to the
   * type specified for that <i>(rule class, attribute name)</i> pair. Errors reported at this stage
   * include: missing value for mandatory attribute, value of wrong type. Such error cause Rule
   * construction to be aborted, so the resulting package will have missing members.
   *
   * @see PackageFactory#PackageFactory
   */
  @VisibleForTesting // used by PackageFactoryApparatus
  public Package.Builder evaluateBuildFile(
      String workspaceName,
      PackageIdentifier packageId,
      BuildFileAST buildFileAST,
      RootedPath buildFilePath,
      Globber globber,
      Iterable<Event> pastEvents,
      Iterable<Postable> pastPosts,
      RuleVisibility defaultVisibility,
      SkylarkSemantics skylarkSemantics,
      Map<String, Extension> imports,
      ImmutableList<Label> skylarkFileDependencies,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws InterruptedException {
    Package.Builder pkgBuilder = new Package.Builder(packageBuilderHelper.createFreshPackage(
        packageId, ruleClassProvider.getRunfilesPrefix()));
    StoredEventHandler eventHandler = new StoredEventHandler();

    try (Mutability mutability = Mutability.create("package %s", packageId)) {
      BazelStarlarkContext starlarkContext =
          new BazelStarlarkContext(
              ruleClassProvider.getToolsRepository(),
              repositoryMapping,
              new SymbolGenerator<>(packageId));
      Environment pkgEnv =
          Environment.builder(mutability)
              .setGlobals(BazelLibrary.GLOBALS)
              .setSemantics(skylarkSemantics)
              .setEventHandler(eventHandler)
              .setImportedExtensions(imports)
              .setStarlarkContext(starlarkContext)
              .build();
      SkylarkUtils.setPhase(pkgEnv, Phase.LOADING);

      pkgBuilder.setFilename(buildFilePath)
          .setDefaultVisibility(defaultVisibility)
          // "defaultVisibility" comes from the command line. Let's give the BUILD file a chance to
          // set default_visibility once, be reseting the PackageBuilder.defaultVisibilitySet flag.
          .setDefaultVisibilitySet(false)
          .setSkylarkFileDependencies(skylarkFileDependencies)
          .setWorkspaceName(workspaceName)
          .setRepositoryMapping(repositoryMapping);

      Event.replayEventsOn(eventHandler, pastEvents);
      for (Postable post : pastPosts) {
        eventHandler.post(post);
      }

      // Stuff that closes over the package context:
      PackageContext context =
          new PackageContext(
              pkgBuilder, globber, eventHandler, ruleFactory.getAttributeContainerFactory());
      buildPkgEnv(pkgEnv, context);

      if (!validatePackageIdentifier(packageId, buildFileAST.getLocation(), eventHandler)) {
        pkgBuilder.setContainsErrors();
      }

      if (buildFileAST.containsErrors()) {
        pkgBuilder.setContainsErrors();
      }

      // TODO(bazel-team): (2009) the invariant "if errors are reported, mark the package
      // as containing errors" is strewn all over this class.  Refactor to use an
      // event sensor--and see if we can simplify the calling code in
      // createPackage().
      if (!buildFileAST.exec(pkgEnv, eventHandler)) {
        pkgBuilder.setContainsErrors();
      }
    }

    pkgBuilder.addPosts(eventHandler.getPosts());
    pkgBuilder.addEvents(eventHandler.getEvents());
    return pkgBuilder;
  }

  // Reports an error and returns false iff package identifier was illegal.
  private static boolean validatePackageIdentifier(
      PackageIdentifier packageId, Location location, ExtendedEventHandler eventHandler) {
    String error = LabelValidator.validatePackageName(packageId.getPackageFragment().toString());
    if (error != null) {
      eventHandler.handle(Event.error(location, error));
      return false; // Invalid package name 'foo'
    }
    return true;
  }

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(PackageFactory.class);
  }

  /** Empty EnvironmentExtension */
  public static class EmptyEnvironmentExtension implements EnvironmentExtension {
    @Override
    public void update(Environment environment) {}

    @Override
    public void updateWorkspace(Environment environment) {}

    @Override
    public Iterable<PackageArgument<?>> getPackageArguments() {
      return ImmutableList.of();
    }

    @Override
    public ImmutableList<BaseFunction> nativeModuleFunctions() {
      return ImmutableList.<BaseFunction>of();
    }
  }
}
