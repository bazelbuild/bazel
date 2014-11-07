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

package com.google.devtools.build.lib.packages;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.NoSuchVariableException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Ident;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.MixedModeFunction;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * The package factory is responsible for constructing Package instances
 * from a BUILD file's abstract syntax tree (AST).  The caller may
 * specify whether the AST is to be retained; unless it is specifically
 * required (e.g. for a BUILD file editor or pretty-printer), it should not be
 * retained as it uses a substantial amount of memory.
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
        Package.LegacyBuilder pkgBuilder, Location location, Object value)
        throws EvalException, ConversionException {
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
        Package.LegacyBuilder pkgBuilder, Location location, T value)
        throws EvalException;
  }

  /**
   * An extension to the global namespace of the BUILD language.
   */
  public interface EnvironmentExtension {
    /**
     * Update the global environment with the identifiers this extension contributes.
     */
    void update(Environment environment, MakeEnvironment.Builder pkgMakeEnv,
        Label buildFileLabel);

    Iterable<PackageArgument<?>> getPackageArguments();
  }

  private static final int EXCLUDE_DIR_DEFAULT = 1;

  private static class DefaultVisibility extends PackageArgument<List<Label>> {
    private DefaultVisibility() {
      super("default_visibility", Type.LABEL_LIST);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultVisibility(getVisibility(value));
    }
  }

  private static class DefaultObsolete extends PackageArgument<Boolean> {
    private DefaultObsolete() {
      super("default_obsolete", Type.BOOLEAN);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        Boolean value) {
      pkgBuilder.setDefaultObsolete(value);
    }
  }

  private static class DefaultTestOnly extends PackageArgument<Boolean> {
    private DefaultTestOnly() {
      super("default_testonly", Type.BOOLEAN);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        Boolean value) {
      pkgBuilder.setDefaultTestonly(value);
    }
  }

  private static class DefaultDeprecation extends PackageArgument<String> {
    private DefaultDeprecation() {
      super("default_deprecation", Type.STRING);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        String value) {
      pkgBuilder.setDefaultDeprecation(value);
    }
  }

  private static class Features extends PackageArgument<List<String>> {
    private Features() {
      super("features", Type.STRING_LIST);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        List<String> value) {
      pkgBuilder.addFeatures(value);
    }
  }

  public static final String PKG_CONTEXT = "$pkg_context";

  private static final Logger LOG = Logger.getLogger(PackageFactory.class.getName());

  private final RuleFactory ruleFactory;
  private final RuleClassProvider ruleClassProvider;

  private final Profiler profiler = Profiler.instance();
  private final boolean retainAsts;
  // TODO(bazel-team): Remove this field - it's not used with Skyframe.
  private final Environment globalEnv;

  private AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls;
  private Preprocessor.Factory preprocessorFactory = Preprocessor.Factory.NullFactory.INSTANCE;

  private final ThreadPoolExecutor threadPool;
  private Map<String, String> platformSetRegexps;

  private final ImmutableList<EnvironmentExtension> environmentExtensions;

  /**
   * Constructs a {@code PackageFactory} instance with the given rule factory,
   * never retains ASTs.
   */
  public PackageFactory(RuleClassProvider ruleClassProvider,
      Map<String, String> platformSetRegexps,
      Iterable<EnvironmentExtension> environmentExtensions) {
    this(ruleClassProvider, platformSetRegexps, environmentExtensions, false);
  }

  /**
   * Constructs a {@code PackageFactory} instance with the given rule factory,
   * never retains ASTs.
   */
  public PackageFactory(RuleClassProvider ruleClassProvider) {
    this(ruleClassProvider, null, ImmutableList.<EnvironmentExtension>of(), false);
  }

  @VisibleForTesting
  public PackageFactory(RuleClassProvider ruleClassProvider,
      EnvironmentExtension environmentExtensions) {
    this(ruleClassProvider, null, ImmutableList.of(environmentExtensions), false);
  }
  /**
   * Constructs a {@code PackageFactory} instance with a specific AST retention
   * policy, glob path translator, and rule factory.
   *
   * @param retainAsts should be {@code true} when the factory should create
   *        {@code Package}s that keep a copy of the {@code BuildFileAST}
   * @see #evaluateBuildFile for details on the ast retention policy
   */
  @VisibleForTesting
  public PackageFactory(RuleClassProvider ruleClassProvider,
      Map<String, String> platformSetRegexps,
      Iterable<EnvironmentExtension> environmentExtensions,
      boolean retainAsts) {
    this.platformSetRegexps = platformSetRegexps;
    this.ruleFactory = new RuleFactory(ruleClassProvider);
    this.ruleClassProvider = ruleClassProvider;
    this.retainAsts = retainAsts;
    globalEnv = newGlobalEnvironment();
    threadPool = new ThreadPoolExecutor(100, 100, 3L, TimeUnit.SECONDS,
        new LinkedBlockingQueue<Runnable>(),
        new ThreadFactoryBuilder().setNameFormat("PackageFactory %d").build());
    // Do not consume threads when not in use.
    threadPool.allowCoreThreadTimeOut(true);
    this.environmentExtensions = ImmutableList.copyOf(environmentExtensions);
  }

  /**
   * Sets the preprocessor used.
   */
  public void setPreprocessorFactory(Preprocessor.Factory preprocessorFactory) {
    this.preprocessorFactory = preprocessorFactory;
  }

 /**
   * Sets the syscalls cache used in globbing.
   */
  public void setSyscalls(AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls) {
    this.syscalls = Preconditions.checkNotNull(syscalls);
  }

  /**
   * Returns the static environment initialized once and shared by all packages
   * created by this factory. No updates occur to this environment once created.
   */
  @VisibleForTesting
  public Environment getEnvironment() {
    return globalEnv;
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

  /****************************************************************************
   * Environment function factories.
   */

  /**
   * Returns a function-value implementing "glob" in the specified package
   * context.
   *
   * @param async if true, start globs in the background but don't block on their completion.
   *        Only use this for heuristic preloading.
   */
  private static Function newGlobFunction(
      final PackageContext originalContext, final boolean async) {
    List<String> params = ImmutableList.of("include", "exclude", "exclude_directories");
    return new MixedModeFunction("glob", params, 1, false) {
        @Override
        public Object call(Object[] namedArguments, FuncallExpression ast, Environment env)
                throws EvalException, ConversionException, InterruptedException {

          // Skylark build extensions need to get the PackageContext from the Environment;
          // async glob functions cannot do the same because the Environment is not thread safe.
          PackageContext context;
          if (originalContext == null) {
            Preconditions.checkArgument(!async);
            try {
              context = (PackageContext) env.lookup(PKG_CONTEXT);
            } catch (NoSuchVariableException e) {
              throw new EvalException(ast.getLocation(), e.getMessage());
            }
          } else {
            context = originalContext;
          }

          List<String> includes = Type.STRING_LIST.convert(namedArguments[0], "'glob' argument");
          List<String> excludes = namedArguments[1] == null
              ? Collections.<String>emptyList()
              : Type.STRING_LIST.convert(namedArguments[1], "'glob' argument");
          int excludeDirs = namedArguments[2] == null
            ? EXCLUDE_DIR_DEFAULT
            : Type.INTEGER.convert(namedArguments[2], "'glob' argument");

          if (async) {
            try {
              context.pkgBuilder.globAsync(includes, excludes, excludeDirs != 0);
            } catch (GlobCache.BadGlobException e) {
              // Ignore: errors will appear during the actual evaluation of the package.
            }
            return GlobList.captureResults(includes, excludes, ImmutableList.<String>of());
          } else {
            return handleGlob(includes, excludes, excludeDirs != 0, context, ast);
          }
        }
      };
  }

  /**
   * Adds a glob to the package, reporting any errors it finds.
   *
   * @param includes the list of includes which must be non-null
   * @param excludes the list of excludes which must be non-null
   * @param context the package context
   * @param ast the AST
   * @return the list of matches
   * @throws EvalException if globbing failed
   */
  private static GlobList<String> handleGlob(List<String> includes, List<String> excludes,
      boolean excludeDirs, PackageContext context, FuncallExpression ast)
        throws EvalException, InterruptedException {
    try {
      List<String> matches = context.pkgBuilder.glob(includes, excludes, excludeDirs);
      return GlobList.captureResults(includes, excludes, matches);
    } catch (IOException expected) {
      context.eventHandler.handle(Event.error(ast.getLocation(),
              "error globbing [" + Joiner.on(", ").join(includes) + "]: " + expected.getMessage()));
      context.pkgBuilder.setContainsTemporaryErrors();
      return GlobList.captureResults(includes, excludes, ImmutableList.<String>of());
    } catch (GlobCache.BadGlobException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  /**
   * Returns a function value implementing the "mocksubinclude" function,
   * emitted by the PythonPreprocessor.  We annotate the
   * package with additional dependencies.  (A 'real' subinclude will never be
   * seen by the parser, because the presence of "subinclude" triggers
   * preprocessing.)
   */
  private static Function newMockSubincludeFunction(final PackageContext context) {
    return new MixedModeFunction("mocksubinclude", ImmutableList.of("label", "path"), 2, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast)
            throws ConversionException {
          Label label = Type.LABEL.convert(args[0], "'mocksubinclude' argument",
                                           context.pkgBuilder.getBuildFileLabel());
          String pathString = Type.STRING.convert(args[1], "'mocksubinclude' argument");
          Path path = pathString.isEmpty()
              ? null
              : context.pkgBuilder.getFilename().getRelative(pathString);
          // A subinclude within a package counts as a file declaration.
          if (label.getPackageIdentifier().equals(context.pkgBuilder.getPackageIdentifier())) {
            Location location = ast.getLocation();
            if (location == null) {
              location = Location.fromFile(context.pkgBuilder.getFilename());
            }
            context.pkgBuilder.createInputFileMaybe(label, location);
          }

          context.pkgBuilder.addSubinclude(label, path);
          return Environment.NONE;
        }
      };
  }

  /**
   * Fake function: subinclude calls are ignored
   * They will disappear after the Python preprocessing.
   */
  private static Function newSubincludeFunction() {
    return new MixedModeFunction("subinclude", ImmutableList.of("file"), 1, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast) {
          return Environment.NONE;
        }
      };
  }

  /**
   * Returns a function-value implementing "exports_files" in the specified
   * package context.
   */
  private static Function newExportsFilesFunction(final PackageContext context) {
    final Package.LegacyBuilder pkgBuilder = context.pkgBuilder;
    List<String> params = ImmutableList.of("srcs", "visibility", "licenses");
    return new MixedModeFunction("exports_files", params, 1, false) {
      @Override
      public Object call(Object[] namedArgs, FuncallExpression ast)
          throws EvalException, ConversionException {

        List<String> files = Type.STRING_LIST.convert(namedArgs[0], "'exports_files' operand");

        RuleVisibility visibility = namedArgs[1] == null
            ? ConstantRuleVisibility.PUBLIC
            : getVisibility(Type.LABEL_LIST.convert(
                namedArgs[1],
                "'exports_files' operand",
                pkgBuilder.getBuildFileLabel()));
        License license = namedArgs[2] == null
            ? null
            : Type.LICENSE.convert(namedArgs[2], "'exports_files' operand");

        for (String file : files) {
          String errorMessage = LabelValidator.validateTargetName(file);
          if (errorMessage != null) {
            throw new EvalException(ast.getLocation(), errorMessage);
          }
          try {
            InputFile inputFile = pkgBuilder.createInputFile(file, ast.getLocation());
            if (inputFile.isVisibilitySpecified() &&
                inputFile.getVisibility() != visibility) {
              throw new EvalException(ast.getLocation(),
                  String.format("visibility for exported file '%s' declared twice",
                      inputFile.getName()));
            }
            if (license != null && inputFile.isLicenseSpecified()) {
              throw new EvalException(ast.getLocation(),
                  String.format("licenses for exported file '%s' declared twice",
                      inputFile.getName()));
            }
            if (license == null && pkgBuilder.getDefaultLicense() == License.NO_LICENSE
                && pkgBuilder.getBuildFileLabel().toString().startsWith("//third_party/")) {
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
        return Environment.NONE;
      }
    };
  }

  /**
   * Returns a function-value implementing "licenses" in the specified package
   * context.
   */
  private static Function newLicensesFunction(final PackageContext context) {
    return new MixedModeFunction("licenses", ImmutableList.of("object"), 1, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast) {
          try {
            License license = Type.LICENSE.convert(args[0], "'licenses' operand");
            context.pkgBuilder.setDefaultLicense(license);
          } catch (ConversionException e) {
            context.eventHandler.handle(Event.error(ast.getLocation(), e.getMessage()));
            context.pkgBuilder.setContainsErrors();
          }
          return Environment.NONE;
        }
      };
  }

  /**
   * Returns a function-value implementing "distribs" in the specified package
   * context.
   */
  private static Function newDistribsFunction(final PackageContext context) {
    return new MixedModeFunction("distribs", ImmutableList.of("object"), 1, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast) {
          try {
            Set<DistributionType> distribs = Type.DISTRIBUTIONS.convert(args[0],
                "'distribs' operand");
            context.pkgBuilder.setDefaultDistribs(distribs);
          } catch (ConversionException e) {
            context.eventHandler.handle(Event.error(ast.getLocation(), e.getMessage()));
            context.pkgBuilder.setContainsErrors();
          }
          return Environment.NONE;
        }
      };
  }

  private static Function newPackageGroupFunction(final PackageContext context) {
    List<String> params = ImmutableList.of("name", "packages", "includes");
    return new MixedModeFunction("package_group", params, 1, true) {
        @Override
        public Object call(Object[] namedArgs, FuncallExpression ast)
            throws EvalException, ConversionException {
          Preconditions.checkState(namedArgs[0] != null);
          String name = Type.STRING.convert(namedArgs[0], "'package_group' argument");
          List<String> packages = namedArgs[1] == null
              ? Collections.<String>emptyList()
              : Type.STRING_LIST.convert(namedArgs[1], "'package_group' argument");
          List<Label> includes = namedArgs[2] == null
              ? Collections.<Label>emptyList()
              : Type.LABEL_LIST.convert(namedArgs[2], "'package_group argument'",
                                        context.pkgBuilder.getBuildFileLabel());

          try {
            context.pkgBuilder.addPackageGroup(name, packages, includes, context.eventHandler,
                ast.getLocation());
            return Environment.NONE;
          } catch (Label.SyntaxException e) {
            throw new EvalException(ast.getLocation(),
                "package group has invalid name: " + name + ": " + e.getMessage());
          } catch (Package.NameConflictException e) {
            throw new EvalException(ast.getLocation(), e.getMessage());
          }
        }
      };
  }

  public static RuleVisibility getVisibility(List<Label> original) {
    RuleVisibility result;

    result = ConstantRuleVisibility.tryParse(original);
    if (result != null) {
      return result;
    }

    result = PackageGroupsRuleVisibility.tryParse(original);
    return result;
  }

  /**
   * Returns a function-value implementing "package" in the specified package
   * context.
   */
  private static Function newPackageFunction(
      final PackageContext context, final Map<String, PackageArgument<?>> packageArguments) {
    return new MixedModeFunction("package", packageArguments.keySet(), 0, true) {
      @Override
      public Object call(Object[] namedArguments, FuncallExpression ast)
          throws EvalException, ConversionException {

        Package.LegacyBuilder pkgBuilder = context.pkgBuilder;

        // Validate parameter list
        if (pkgBuilder.isPackageFunctionUsed()) {
          throw new EvalException(ast.getLocation(),
              "'package' can only be used once per BUILD file");
        }
        pkgBuilder.setPackageFunctionUsed();

        // Parse params
        boolean foundParameter = false;

        int argNumber = 0;
        for (Map.Entry<String, PackageArgument<?>> entry : packageArguments.entrySet()) {
          Object arg = namedArguments[argNumber];
          argNumber += 1;
          if (arg == null) {
            continue;
          }

          foundParameter = true;
          entry.getValue().convertAndProcess(pkgBuilder, ast.getLocation(), arg);
        }

        if (!foundParameter) {
          throw new EvalException(ast.getLocation(),
              "at least one argument must be given to the 'package' function");
        }

        return Environment.NONE;
      }
    };
  }

  // Helper function for createRuleFunction.
  private static void addRule(RuleFactory ruleFactory,
                              String ruleClassName,
                              PackageContext context,
                              Map<String, Object> kwargs,
                              FuncallExpression ast)
      throws RuleFactory.InvalidRuleException, Package.NameConflictException {
    RuleClass ruleClass = getBuiltInRuleClass(ruleClassName, ruleFactory);
    RuleFactory.createAndAddRule(context, ruleClass, kwargs, ast);
  }

  private static RuleClass getBuiltInRuleClass(String ruleClassName, RuleFactory ruleFactory) {
    if (ruleFactory.getRuleClassNames().contains(ruleClassName)) {
      return ruleFactory.getRuleClass(ruleClassName);
    }
    throw new IllegalArgumentException("no such rule class: "  + ruleClassName);
  }

  /**
   * Returns a function-value implementing the build rule "ruleClass" (e.g. cc_library) in the
   * specified package context.
   */
  private static Function newRuleFunction(final RuleFactory ruleFactory,
                                          final String ruleClass) {
    return new AbstractFunction(ruleClass) {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env)
          throws EvalException {
        if (!args.isEmpty()) {
          throw new EvalException(ast.getLocation(),
              "build rules do not accept positional parameters");
        }

        try {
          PackageContext context = (PackageContext) env.lookup(PKG_CONTEXT);
          addRule(ruleFactory, ruleClass, context, kwargs, ast);
        } catch (RuleFactory.InvalidRuleException | Package.NameConflictException
            | NoSuchVariableException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
        return Environment.NONE;
      }
    };
  }

  /**
   * Returns a new environment populated with common entries that can be shared
   * across packages and that don't require the context.
   */
  private static Environment newGlobalEnvironment() {
    Environment env = new Environment();
    MethodLibrary.setupMethodEnvironment(env);
    return env;
  }

  /****************************************************************************
   * Package creation.
   */

  /**
   * Loads, scans parses and evaluates the build file at "buildFile", and
   * creates and returns a Package builder instance capable of building a package identified by
   * "packageId".
   *
   * <p>This method returns a builder to allow the caller to do additional work, if necessary.
   *
   * <p>This method assumes "packageId" is a valid package name according to the
   * {@link LabelValidator#validatePackageName} heuristic.
   *
   * <p>This method allows the caller to inject build file contents by
   * specifying the {@code replacementSource} parameter. If {@code null}, the
   * contents are loaded from the {@code buildFile}.
   *
   * <p>See {@link #evaluateBuildFile} for information on AST retention.
   */
  public Package.LegacyBuilder createPackage(PackageIdentifier packageId, Path buildFile,
      List<Statement> preludeStatements, ParserInputSource inputSource,
      Map<PathFragment, SkylarkEnvironment> imports,
      CachingPackageLocator locator, ParserInputSource replacementSource,
      RuleVisibility defaultVisibility) throws InterruptedException {
    profiler.startTask(ProfilerTask.CREATE_PACKAGE, packageId.toString());
    StoredEventHandler localReporter = new StoredEventHandler();
    GlobCache globCache = createGlobCache(buildFile.getParentDirectory(), packageId, locator);
    try {
      // Run the lexer and parser with a local reporter, so that errors from other threads do not
      // show up below. Merge the local and global reporters afterwards.
      // Logged message is used as a testability hook tracing the parsing progress
      LOG.fine("Starting to parse " + packageId);

      BuildFileAST buildFileAST;
      boolean hasPreprocessorError = false;
      // TODO(bazel-team): It would be nicer to always pass in the right value rather than rely
      // on the null value.
      Preprocessor.Result preprocessingResult = replacementSource == null
          ? getParserInput(packageId, buildFile, inputSource, globCache, localReporter)
          : Preprocessor.Result.success(replacementSource, false);
      if (localReporter.hasErrors()) {
        hasPreprocessorError = true;
      }

      buildFileAST = BuildFileAST.parseBuildFile(
          preprocessingResult.result, preludeStatements, localReporter, locator, false);
      // Logged message is used as a testability hook tracing the parsing progress
      LOG.fine("Finished parsing of " + packageId);

      MakeEnvironment.Builder makeEnv = new MakeEnvironment.Builder();
      if (platformSetRegexps != null) {
        makeEnv.setPlatformSetRegexps(platformSetRegexps);
      }

      // At this point the package is guaranteed to exist.  It may have parse or
      // evaluation errors, resulting in a diminished number of rules.
      prefetchGlobs(packageId, buildFileAST, preprocessingResult.preprocessed,
          buildFile, globCache, defaultVisibility, makeEnv);

      return evaluateBuildFile(
          packageId, buildFileAST, buildFile, globCache, localReporter.getEvents(),
          defaultVisibility, hasPreprocessorError, preprocessingResult.containsTransientErrors,
          makeEnv, imports);
    } catch (InterruptedException e) {
      globCache.cancelBackgroundTasks();
      throw e;
    } finally {
      globCache.finishBackgroundTasks();
      profiler.completeTask(ProfilerTask.CREATE_PACKAGE);
    }
  }

  /**
   * Same as {@link #createPackage}, but does the required validation of "packageName" first,
   * throwing a {@link NoSuchPackageException} if the name is invalid.
   */
  @VisibleForTesting
  public Package createPackageForTesting(PackageIdentifier packageId, Path buildFile,
      CachingPackageLocator locator, EventHandler eventHandler)
          throws NoSuchPackageException, InterruptedException {
    String error = LabelValidator.validatePackageName(
        packageId.getPackageFragment().getPathString());
    if (error != null) {
      throw new BuildFileNotFoundException(packageId.toString(),
          "illegal package name: '" + packageId.toString() + "' (" + error + ")");
    }
    ParserInputSource inputSource = getParserInputSource(buildFile, eventHandler);
    if (inputSource == null) {
      throw new BuildFileContainsErrorsException(packageId.toString(), "IOException occured");
    }
    Package result = createPackage(packageId, buildFile,
        ImmutableList.<Statement>of(), inputSource,
        ImmutableMap.<PathFragment, SkylarkEnvironment>of(),
        locator, null, ConstantRuleVisibility.PUBLIC).build();
    Event.replayEventsOn(eventHandler, result.getEvents());
    return result;
  }

  /**
   * Returns the parser input (with preprocessing already applied, if
   * applicable) for the specified package and build file.
   *
   * @param packageId the identifier for the package; used for error messages
   * @param buildFile the path of the BUILD file to read
   * @param locator package locator used in recursive globbing
   * @param eventHandler the eventHandler on which preprocessing errors/warnings are to
   *        be reported
   * @throws NoSuchPackageException if the build file cannot be read
   * @return the preprocessed input, as seen by Blaze's parser
   */
  // Used externally!
  public ParserInputSource getParserInput(PackageIdentifier packageId, Path buildFile,
      CachingPackageLocator locator, EventHandler eventHandler)
      throws NoSuchPackageException, InterruptedException {
    ParserInputSource inputSource = getParserInputSource(buildFile, eventHandler);
    if (inputSource == null) {
      return Preprocessor.Result.transientError(buildFile).result;
    }
    return getParserInput(
        packageId, buildFile, getParserInputSource(buildFile, eventHandler),
        createGlobCache(buildFile.getParentDirectory(), packageId, locator),
        eventHandler).result;
  }

  private GlobCache createGlobCache(Path packageDirectory, PackageIdentifier packageId,
      CachingPackageLocator locator) {
    return new GlobCache(packageDirectory, packageId, locator, syscalls, threadPool);
  }

  @Nullable private ParserInputSource getParserInputSource(
      Path buildFile, EventHandler eventHandler) {
    ParserInputSource inputSource;
    try {
      inputSource = ParserInputSource.create(buildFile);
    } catch (IOException e) {
      eventHandler.handle(Event.error(Location.fromFile(buildFile), e.getMessage()));
      return null;
    }
    return inputSource;
  }

  /**
   * Version of #getParserInput(String, Path, GlobCache, Reporter) that allows
   * to inject a glob cache that gets populated during preprocessing.
   */
  private Preprocessor.Result getParserInput(
      PackageIdentifier packageId, Path buildFile, ParserInputSource inputSource,
      GlobCache globCache, EventHandler eventHandler)
          throws InterruptedException {
    Preprocessor preprocessor = preprocessorFactory.getPreprocessor();
    if (preprocessor == null) {
      return Preprocessor.Result.success(inputSource, false);
    }

    try {
      return preprocessor.preprocess(inputSource, packageId.toString(), globCache, eventHandler,
          globalEnv, ruleFactory.getRuleClassNames());
    } catch (IOException e) {
      eventHandler.handle(Event.error(Location.fromFile(buildFile),
                     "preprocessing failed: " + e.getMessage()));
      return Preprocessor.Result.transientError(buildFile);
    }
  }

  /**
   * This tuple holds the current package builder, current lexer, etc, for the
   * duration of the evaluation of one BUILD file. (We use a PackageContext
   * object in preference to storing these values in mutable fields of the
   * PackageFactory.)
   *
   * <p>PLEASE NOTE: references to PackageContext objects are held by many
   * Function closures, but should become unreachable once the Environment is
   * discarded at the end of evaluation.  Please be aware of your memory
   * footprint when making changes here!
   */
  public static class PackageContext {

    final Package.LegacyBuilder pkgBuilder;
    final EventHandler eventHandler;
    final boolean retainASTs;

    public PackageContext(Package.LegacyBuilder pkgBuilder, EventHandler eventHandler,
        boolean retainASTs) {
      this.pkgBuilder = pkgBuilder;
      this.eventHandler = eventHandler;
      this.retainASTs = retainASTs;
    }
  }

  private void buildPkgEnv(
      Environment pkgEnv, String packageName, PackageContext context) {
    ImmutableList.Builder<PackageArgument<?>> arguments =
        ImmutableList.<PackageArgument<?>>builder()
           .add(new DefaultVisibility())
           .add(new DefaultDeprecation())
           .add(new DefaultObsolete())
           .add(new DefaultTestOnly())
           .add(new Features());


    for (EnvironmentExtension extension : environmentExtensions) {
      arguments.addAll(extension.getPackageArguments());
    }

    ImmutableMap.Builder<String, PackageArgument<?>> packageArguments = ImmutableMap.builder();
    for (PackageArgument<?> argument : arguments.build()) {
      packageArguments.put(argument.getName(), argument);
    }
    pkgEnv.update("distribs", newDistribsFunction(context));
    pkgEnv.update("glob", newGlobFunction(context, /*async=*/false));
    pkgEnv.update("mocksubinclude", newMockSubincludeFunction(context));
    pkgEnv.update("licenses", newLicensesFunction(context));
    pkgEnv.update("exports_files", newExportsFilesFunction(context));
    pkgEnv.update("package_group", newPackageGroupFunction(context));
    pkgEnv.update("package", newPackageFunction(context, packageArguments.build()));
    pkgEnv.update("subinclude", newSubincludeFunction());

    pkgEnv.update("PACKAGE_NAME", packageName);
  }

  /**
   * Returns the list of native rule functions created using the {@link RuleClassProvider}
   * of this {@link PackageFactory}.
   */
  public ImmutableList<Function> collectNativeRuleFunctions() {
    ImmutableList.Builder<Function> builder = ImmutableList.builder();
    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      builder.add(newRuleFunction(ruleFactory, ruleClass));
    }
    builder.add(newGlobFunction(null, false));
    return builder.build();
  }

  private void buildPkgEnv(Environment pkgEnv, String packageName,
      MakeEnvironment.Builder pkgMakeEnv, PackageContext context, RuleFactory ruleFactory) {
    buildPkgEnv(pkgEnv, packageName, context);
    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      Function ruleFunction = newRuleFunction(ruleFactory, ruleClass);
      pkgEnv.update(ruleClass, ruleFunction);
    }

    for (EnvironmentExtension extension : environmentExtensions) {
      extension.update(pkgEnv, pkgMakeEnv, context.pkgBuilder.getBuildFileLabel());
    }
  }

  /**
   * Constructs a Package instance, evaluates the BUILD-file AST inside the
   * build environment, and populates the package with Rule instances as it
   * goes.  As with most programming languages, evaluation stops when an
   * exception is encountered: no further rules after the point of failure will
   * be constructed.  We assume that rules constructed before the point of
   * failure are valid; this assumption is not entirely correct, since a
   * "vardef" after a rule declaration can affect the behavior of that rule.
   *
   * <p>Rule attribute checking is performed during evaluation. Each attribute
   * must conform to the type specified for that <i>(rule class, attribute
   * name)</i> pair.  Errors reported at this stage include: missing value for
   * mandatory attribute, value of wrong type.  Such error cause Rule
   * construction to be aborted, so the resulting package will have missing
   * members.
   *
   * <p>If the factory is created with {@code true} for the {@code retainAsts}
   * parameter, the {@code Package} returned from this method will
   * contain a {@link BuildFileAST} when calling {@link
   * Package#getSyntaxTree()},  otherwise it will return {@code null}.
   *
   * @see Package#getSyntaxTree()
   * @see PackageFactory#PackageFactory
   */
  @VisibleForTesting // used by PackageFactoryApparatus
  public Package.LegacyBuilder evaluateBuildFile(PackageIdentifier packageId,
      BuildFileAST buildFileAST, Path buildFilePath, GlobCache globCache,
      Iterable<Event> pastEvents, RuleVisibility defaultVisibility, boolean containsError,
      boolean containsTransientError, MakeEnvironment.Builder pkgMakeEnv,
      Map<PathFragment, SkylarkEnvironment> imports) throws InterruptedException {
    // Important: Environment should be unreachable by the end of this method!
    Environment pkgEnv = new Environment(globalEnv);

    Package.LegacyBuilder pkgBuilder =
        new Package.LegacyBuilder(packageId)
        .setFilename(buildFilePath)
        .setAST(retainAsts ? buildFileAST : null)
        .setMakeEnv(pkgMakeEnv)
        .setGlobCache(globCache)
        .setDefaultVisibility(defaultVisibility)
        // "defaultVisibility" comes from the command line. Let's give the BUILD file a chance to
        // set default_visibility once, be reseting the PackageBuilder.defaultVisibilitySet flag.
        .setDefaultVisibilitySet(false);

    StoredEventHandler eventHandler = new StoredEventHandler();
    Event.replayEventsOn(eventHandler, pastEvents);

    // Stuff that closes over the package context:
    PackageContext context = new PackageContext(pkgBuilder, eventHandler, retainAsts);
    buildPkgEnv(pkgEnv, packageId.toString(), pkgMakeEnv, context, ruleFactory);

    if (containsError) {
      pkgBuilder.setContainsErrors();
    }

    if (containsTransientError) {
      pkgBuilder.setContainsTemporaryErrors();
    }

    if (!validatePackageIdentifier(packageId, buildFileAST.getLocation(), eventHandler)) {
      pkgBuilder.setContainsErrors();
    }

    pkgEnv.setImportedExtensions(imports);
    pkgEnv.updateAndPropagate(PKG_CONTEXT, context);
    pkgEnv.updateAndPropagate(Environment.PKG_NAME, packageId.toString());

    if (!validateAssignmentStatements(pkgEnv, buildFileAST, eventHandler)) {
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

    pkgBuilder.addEvents(eventHandler.getEvents());
    return pkgBuilder;
  }

  /**
   * Visit all targets and expand the globs in parallel.
   */
  private void prefetchGlobs(PackageIdentifier packageId, BuildFileAST buildFileAST,
      boolean wasPreprocessed, Path buildFilePath, GlobCache globCache,
      RuleVisibility defaultVisibility, MakeEnvironment.Builder pkgMakeEnv)
      throws InterruptedException {
    if (wasPreprocessed) {
      // No point in prefetching globs here: preprocessing implies eager evaluation
      // of all globs.
      return;
    }
    // Important: Environment should be unreachable by the end of this method!
    Environment pkgEnv = new Environment();

    Package.LegacyBuilder pkgBuilder =
        new Package.LegacyBuilder(packageId)
        .setFilename(buildFilePath)
        .setMakeEnv(pkgMakeEnv)
        .setGlobCache(globCache)
        .setDefaultVisibility(defaultVisibility)
        // "defaultVisibility" comes from the command line. Let's give the BUILD file a chance to
        // set default_visibility once, be reseting the PackageBuilder.defaultVisibilitySet flag.
        .setDefaultVisibilitySet(false);

    // Stuff that closes over the package context:
    PackageContext context = new PackageContext(pkgBuilder, NullEventHandler.INSTANCE, false);
    buildPkgEnv(pkgEnv, packageId.toString(), context);
    pkgEnv.update("glob", newGlobFunction(context, /*async=*/true));
    // The Fileset function is heavyweight in that it can run glob(). Avoid this during the
    // preloading phase.
    pkgEnv.remove("FilesetEntry");

    buildFileAST.exec(pkgEnv, NullEventHandler.INSTANCE);
  }


  /**
   * Tests a build AST to ensure that it contains no assignment statements that
   * redefine built-in build rules.
   *
   * @param pkgEnv a package environment initialized with all of the built-in
   *        build rules
   * @param ast the build file AST to be tested
   * @param eventHandler a eventHandler where any errors should be logged
   * @return true if the build file contains no redefinitions of built-in
   *         functions
   */
  private static boolean validateAssignmentStatements(Environment pkgEnv,
                                                      BuildFileAST ast,
                                                      EventHandler eventHandler) {
    for (Statement stmt : ast.getStatements()) {
      if (stmt instanceof AssignmentStatement) {
        Expression lvalue = ((AssignmentStatement) stmt).getLValue();
        if (!(lvalue instanceof Ident)) {
          continue;
        }
        String target = ((Ident) lvalue).getName();
        if (pkgEnv.lookup(target, null) != null) {
          eventHandler.handle(Event.error(stmt.getLocation(), "Reassignment of builtin build "
              + "function '" + target + "' not permitted"));
          return false;
        }
      }
    }
    return true;
  }

  // Reports an error and returns false iff package identifier was illegal.
  private static boolean validatePackageIdentifier(PackageIdentifier packageId, Location location,
      EventHandler eventHandler) {
    String error = LabelValidator.validateWorkspaceName(packageId.getRepository());
    if (error != null) {
      eventHandler.handle(Event.error(location, error));
      return false; // Invalid package repo '@foo'
    }
    error = LabelValidator.validatePackageName(packageId.getPackageFragment().toString());
    if (error != null) {
      eventHandler.handle(Event.error(location, error));
      return false; // Invalid package name 'foo'
    }
    return true;
  }
}
