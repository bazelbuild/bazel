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
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullErrorEventListener;
import com.google.devtools.build.lib.events.StoredErrorEventListener;
import com.google.devtools.build.lib.packages.LegacyPackage.LegacyPackageBuilder;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Ident;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.MixedModeFunction;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.PositionalFunction;
import com.google.devtools.build.lib.syntax.SelectorValue;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
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

  private static final int EXCLUDE_DIR_DEFAULT = 1;

  /**
   * Positional parameters for the 'package' function.
   */
  private static enum PackageParams {
    DEFAULT_VISIBILITY("default_visibility"),
    DEFAULT_OBSOLETE("default_obsolete"),
    DEFAULT_DEPRECATION("default_deprecation"),
    DEFAULT_TESTONLY("default_testonly"),
    FEATURES("features");

    private final String name;

    PackageParams(String name) {
      this.name = name;
    }

    public static Iterable<String> getParams() {
      return params;
    }

    private static final ImmutableList<String> params = getAllParams();

    private static ImmutableList<String> getAllParams() {
      ImmutableList.Builder<String> builder = ImmutableList.builder();
      for (PackageParams param : PackageParams.values()) {
        builder.add(param.name);
      }
      return builder.build();
    }
  }

  private static final Logger LOG = Logger.getLogger(PackageFactory.class.getName());

  private final RuleFactory ruleFactory;
  private final SkylarkRuleFactory skylarkRuleFactory;
  private final RuleClassProvider ruleClassProvider;

  private final Profiler profiler = Profiler.instance();
  private final boolean retainAsts;
  // TODO(bazel-team): Remove this field - it's not used with Skyframe.
  private final boolean trackFsDeps;
  private final Environment globalEnv;

  private AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls;
  private Preprocessor preprocessor;
  private boolean skylarkEnabled;

  private final ThreadPoolExecutor threadPool;
  private Map<String, String> platformSetRegexps;

  /**
   * Constructs a {@code PackageFactory} instance with the given rule factory,
   * never retains ASTs.
   */
  // TODO(bazel-team): Remove the trackFsDeps parameter.
  public PackageFactory(RuleClassProvider ruleClassProvider, boolean trackFsDeps,
      Map<String, String> platformSetRegexps) {
    this(false, ruleClassProvider, trackFsDeps, platformSetRegexps);
  }

  /**
   * Constructs a {@code PackageFactory} instance with the given rule factory,
   * never retains ASTs.
   */
  public PackageFactory(RuleClassProvider ruleClassProvider, boolean trackFsDeps) {
    this(false, ruleClassProvider, trackFsDeps, null);
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
  public PackageFactory(boolean retainAsts, RuleClassProvider ruleClassProvider,
      boolean trackFsDeps, Map<String, String> platformSetRegexps) {
    this.platformSetRegexps = platformSetRegexps;
    this.ruleFactory = new RuleFactory(ruleClassProvider);
    this.skylarkRuleFactory = new SkylarkRuleFactory(ruleClassProvider);
    this.ruleClassProvider = ruleClassProvider;
    this.retainAsts = retainAsts;
    globalEnv = newGlobalEnvironment();
    threadPool = new ThreadPoolExecutor(100, 100, 3L, TimeUnit.SECONDS,
        new LinkedBlockingQueue<Runnable>(),
        new ThreadFactoryBuilder().setNameFormat("PackageFactory %d").build());
    // Do not consume threads when not in use.
    threadPool.allowCoreThreadTimeOut(true);
    this.trackFsDeps = trackFsDeps;
  }

  /**
   * Sets the preprocessor used.
   */
  public void setPreprocessor(Preprocessor preprocessor) {
    this.preprocessor = preprocessor;
  }

 /**
   * Sets the syscalls cache used in globbing.
   */
  public void setSyscalls(AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls) {
    this.syscalls = Preconditions.checkNotNull(syscalls);
  }

 /**
   * Sets whether Skylark import statements are allowed.
   */
  public void setSkylarkEnabled(boolean skylarkEnabled) {
    this.skylarkEnabled = skylarkEnabled;
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
  private static Function newGlobFunction(final PackageContext context, final boolean async) {
    List<String> params = ImmutableList.of("include", "exclude", "exclude_directories");
    return new MixedModeFunction("glob", params, 1, false) {
        @Override
        public Object call(Object[] namedArguments,
            List<Object> surplusPositionalArguments,
            Map<String, Object> surplusKeywordArguments,
            FuncallExpression ast)
                throws EvalException, ConversionException, InterruptedException {
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
      context.listener.error(ast.getLocation(),
          "error globbing [" + Joiner.on(", ").join(includes) + "]: " + expected.getMessage());
      context.pkgBuilder.setContainsTemporaryErrors();
      return GlobList.captureResults(includes, excludes, ImmutableList.<String>of());
    } catch (GlobCache.BadGlobException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  /**
   * Returns a function-value implementing "select" (i.e. configurable attributes)
   * in the specified package context.
   */
  private static Function newSelectFunction() {
    return new PositionalFunction("select", 1, 1) {
      @Override
      public Object call(List<Object> args, FuncallExpression ast)
          throws EvalException, ConversionException {
        Object dict = Iterables.getOnlyElement(args);
        if (!(dict instanceof Map<?, ?>)) {
          throw new EvalException(ast.getLocation(),
              "select({...}) argument isn't a dictionary");
        }
        return new SelectorValue((Map<?, ?>) dict);
      }
    };
  }

  /**
   * Returns a function-value implementing "FilesetEntry" in the specified package
   * context.
   */
  private static Function newFilesetEntryFunction(final PackageContext context) {
    List<String> params = ImmutableList.of("srcdir", "files", "destdir",
        "excludes", "symlinks", "strip_prefix");
    return new MixedModeFunction("FilesetEntry", params, 0, true) {
      @Override
      public Object call(Object[] namedArguments,
          List<Object> surplusPositionalArguments,
          Map<String, Object> surplusKeywordArguments,
          FuncallExpression ast) throws EvalException, ConversionException, InterruptedException {

        final String filesetArg = "'FilesetEntry' argument";

        Label srcLabel = null;
        String srcDir = namedArguments[0] == null
            ? null
            : Type.STRING.convert(namedArguments[0], filesetArg);

        // The srcDir can either refer to a package as whole, or another Fileset rule.
        // If the former, we set the label to be the package's BUILD file.
        if (srcDir == null) {
          // If srcDir is null, it refers to the current package.
          srcLabel = context.pkgBuilder.getBuildFileLabel();
        } else if (srcDir.startsWith("//") && !srcDir.contains(":")) {
          // Here, srcDir refers to a remote package.
          try {
            srcLabel = Label.parseAbsolute(srcDir + ":BUILD");
          } catch (Label.SyntaxException e) {
            throw new EvalException(ast.getLocation(),
                "Could not parse 'srcDir' label '" + srcDir +
                "' in FilesetEntry: " + e.getMessage());
          }
        } else {
          // Here, srcDir refers to an arbitrary label, though we must assert
          // during analysis that it refers to another Fileset.
          srcLabel = Type.LABEL.convert(srcDir, filesetArg,
              context.pkgBuilder.getBuildFileLabel());
        }

        List<Label> files;
        if (namedArguments[1] == null) {
          files = null;
        } else {
          files =
              Type.LABEL_LIST.convert(namedArguments[1], filesetArg, srcLabel);

          for (Label label : files) {
            if (!srcLabel.getPackageFragment().equals(label.getPackageFragment())) {
              throw new EvalException(ast.getLocation(), "'files' element '" + label +
                                                         "' must be a relative label.");
            }
          }
        }
        String destDir = namedArguments[2] == null
            ? null
            : Type.STRING.convert(namedArguments[2], filesetArg);

        List<String> excludes = namedArguments[3] == null
            ? null
            : Type.STRING_LIST.convert(namedArguments[3], filesetArg, srcLabel);

        FilesetEntry.SymlinkBehavior symlinkBehavior = FilesetEntry.SymlinkBehavior.COPY;
        if (namedArguments[4] != null) {
          String behaviorString =
            Type.STRING.convert(namedArguments[4], filesetArg);

          try {
            symlinkBehavior =
              FilesetEntry.SymlinkBehavior.parse(behaviorString);
          } catch (IllegalArgumentException e) {
            throw new EvalException(ast.getLocation(),
                "Invalid value for 'symlinks' attribute. Valid values are " +
                "'copy' and 'dereference'.");
          }
        }

        String stripPrefix = namedArguments[5] == null
            ? "."
            : Type.STRING.convert(namedArguments[5], filesetArg);


        return new FilesetEntry(srcLabel, files, excludes, destDir,
                                symlinkBehavior, stripPrefix);
      }
    };
  }

  /**
   * Returns a function value implementing the "mocksubinclude" function,
   * emitted by the PythonPreprocessor.  We annotate the
   * package with additional dependencies.  (A 'real' subinclude will never be
   * seen by the parser, because the presence of "subinclude" triggers
   * preprocessing.)
   */
  private static Function newMockSubincludeFunction(final PackageContext context) {
    return new PositionalFunction("mocksubinclude", 2, 2) {
        @Override
        public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
          Label label = Type.LABEL.convert(args.get(0), "'mocksubinclude' argument",
                                           context.pkgBuilder.getBuildFileLabel());
          String pathString = Type.STRING.convert(args.get(1), "'mocksubinclude' argument");
          Path path = pathString.isEmpty()
              ? null
              : context.pkgBuilder.getFilename().getRelative(pathString);
          // A subinclude within a package counts as a file declaration.
          if (label.getPackageFragment().equals(context.pkgBuilder.getNameFragment())) {
            Location location = ast.getLocation();
            if (location == null) {
              location = Location.fromFile(context.pkgBuilder.getFilename());
            }
            context.pkgBuilder.createInputFileMaybe(label, location);
          }

          context.pkgBuilder.addSubinclude(label, path);
          return 0;
        }
      };
  }

  /**
   * Fake function: subinclude calls are ignored
   * They will disappear after the Python preprocessing.
   */
  private static Function newSubincludeFunction() {
    return new PositionalFunction("subinclude", 1, 1) {
        @Override
        public Object call(List<Object> args, FuncallExpression ast) {
          return 0;
        }
      };
  }

  /**
   * Returns a function-value implementing "exports_files" in the specified
   * package context.
   */
  private static Function newExportsFilesFunction(final PackageContext context) {
    final LegacyPackageBuilder pkgBuilder = context.pkgBuilder;
    List<String> params = ImmutableList.of("srcs", "visibility", "licenses");
    return new MixedModeFunction("exports_files", params, 1, false) {
      @Override
      public Object call(Object[] namedArgs, List<Object> surplusPositionalArguments,
          Map<String, Object> surplusKeywordArguments, FuncallExpression ast)
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
          } catch (Package.PackageBuilder.GeneratedLabelConflict e) {
            throw new EvalException(ast.getLocation(), e.getMessage());
          }
        }
        return 0;
      }
    };
  }

  /**
   * Returns a function-value implementing "licenses" in the specified package
   * context.
   */
  private static Function newLicensesFunction(final PackageContext context) {
    return new PositionalFunction("licenses", 1, 1) {
        @Override
        public Object call(List<Object> args, FuncallExpression ast) {
          try {
            License license = Type.LICENSE.convert(args.get(0), "'licenses' operand");
            context.pkgBuilder.setDefaultLicense(license);
          } catch (ConversionException e) {
            context.listener.error(ast.getLocation(), e.getMessage());
            context.pkgBuilder.setContainsErrors();
          }
          return null;
        }
      };
  }

  /**
   * Returns a function-value implementing "distribs" in the specified package
   * context.
   */
  private static Function newDistribsFunction(final PackageContext context) {
    return new PositionalFunction("distribs", 1, 1) {
        @Override
        public Object call(List<Object> args, FuncallExpression ast) {
          try {
            Set<DistributionType> distribs = Type.DISTRIBUTIONS.convert(args.get(0),
                "'distribs' operand");
            context.pkgBuilder.setDefaultDistribs(distribs);
          } catch (ConversionException e) {
            context.listener.error(ast.getLocation(), e.getMessage());
            context.pkgBuilder.setContainsErrors();
          }
          return null;
        }
      };
  }

  private static Function newPackageGroupFunction(final PackageContext context) {
    List<String> params = ImmutableList.of("name", "packages", "includes");
    return new MixedModeFunction("package_group", params, 1, true) {
        @Override
        public Object call(Object[] namedArgs,
            List<Object> surplusPositionalArguments,
            Map<String, Object> surplusKeywordArguments,
            FuncallExpression ast) throws EvalException, ConversionException {
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
            context.pkgBuilder.addPackageGroup(name, packages, includes, context.listener,
                ast.getLocation());
            return null;
          } catch (Label.SyntaxException e) {
            throw new EvalException(ast.getLocation(),
                "package group has invalid name: " + name + ": " + e.getMessage());
          } catch (Package.PackageBuilder.NameConflictException e) {
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
  private static Function newPackageFunction(final PackageContext context) {
    return new MixedModeFunction("package", PackageParams.getParams(), 0, true) {
      @Override
      public Object call(Object[] namedArguments,
          List<Object> surplusPositionalArguments,
          Map<String, Object> surplusKeywordArguments,
          FuncallExpression ast) throws EvalException, ConversionException {

        LegacyPackageBuilder pkgBuilder = context.pkgBuilder;

        // Validate parameter list
        if (pkgBuilder.isPackageFunctionUsed()) {
          throw new EvalException(ast.getLocation(),
              "'package' can only be used once per BUILD file");
        }
        pkgBuilder.setPackageFunctionUsed();

        // Parse params
        boolean foundParameter = false;
        Label buildFileLabel = pkgBuilder.getBuildFileLabel();
        for (PackageParams param : PackageParams.values()) {
          Object arg = namedArguments[param.ordinal()];
          if (arg == null) {
            continue;
          }
          foundParameter = true;
          switch(param) {
            case DEFAULT_VISIBILITY:
              pkgBuilder.setDefaultVisibility(getVisibility(
                  Type.LABEL_LIST.convert(arg, "'package' argument", buildFileLabel)));
              break;
            case DEFAULT_DEPRECATION:
              pkgBuilder.setDefaultDeprecation(Type.STRING.convert(arg, "'package' argument"));
              break;
            case DEFAULT_OBSOLETE:
              pkgBuilder.setDefaultObsolete(Type.BOOLEAN.convert(arg, "'package' argument"));
              break;
            case DEFAULT_TESTONLY:
              pkgBuilder.setDefaultTestonly(Type.BOOLEAN.convert(arg, "'package' argument"));
              break;
            case FEATURES:
              pkgBuilder.setFeatures(Type.STRING_LIST.convert(arg, "'package argument"));
              break;
            default:
              throw new IllegalStateException("missing implementation for '" + param + "'");
          }
        }

        if (!foundParameter) {
          throw new EvalException(ast.getLocation(),
              "at least one argument must be given to the 'package' function");
        }

        return null;
      }
    };
  }

  // Helper function for createRuleFunction.
  private static Rule addRule(RuleFactory ruleFactory,
                              SkylarkRuleFactory skylarkRuleFactory,
                              String ruleClassName,
                              Path extensionFile,
                              PackageContext context,
                              Map<String, Object> kwargs,
                              FuncallExpression ast)
      throws RuleFactory.InvalidRuleException, Package.PackageBuilder.NameConflictException {
    RuleClass ruleClass = getSkylarkOrBuiltInRuleClass(
        ruleClassName, extensionFile, ruleFactory, skylarkRuleFactory);
    Rule rule = RuleFactory.createRule(context.pkgBuilder, ruleClass, kwargs,
                                       context.listener, ast,
                                       context.retainASTs,
                                       ast.getLocation());
    context.pkgBuilder.addRule(rule);
    return rule;
  }

  private static RuleClass getSkylarkOrBuiltInRuleClass(String ruleClassName, Path file,
      RuleFactory ruleFactory, SkylarkRuleFactory skylarkRuleFactory) {
    if (ruleFactory.getRuleClassNames().contains(ruleClassName)) {
      return ruleFactory.getRuleClass(ruleClassName);
    }
    if (skylarkRuleFactory.hasRuleClass(ruleClassName, file)) {
      return skylarkRuleFactory.getRuleClass(ruleClassName, file);
    }
    throw new IllegalArgumentException("no such rule class: "  + ruleClassName);
  }

  /**
   * Returns a function-value implementing the build rule "ruleClass" (e.g. cc_library) in the
   * specified package context.
   */
  private static Function newRuleFunction(final RuleFactory ruleFactory,
                                          final SkylarkRuleFactory skylarkRuleFactory,
                                          final String ruleClass,
                                          final Path extensionFile,
                                          final PackageContext context) {
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
          addRule(ruleFactory, skylarkRuleFactory, ruleClass, extensionFile, context, kwargs, ast);
        } catch (
            RuleFactory.InvalidRuleException | Package.PackageBuilder.NameConflictException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
        return 0;
      }
    };
  }

  /**
   * Returns a function (closed over the specified MakeEnvironment) which
   * implements "vardef".
   */
  static Function newVarDefFunction(final MakeEnvironment.Builder makeEnv) {
    List<String> params = ImmutableList.of("name", "value", "platform_name");
    return new MixedModeFunction("vardef", params, 2, false) {
        // vardef(varname, value, platformName=null)
        @Override
        public Object call(Object[] namedArguments,
                           List<Object> surplusPositionalArguments,
                           Map<String, Object> surplusKeywordArguments,
                           FuncallExpression ast) throws EvalException, ConversionException {
          String varname = Type.STRING.convert(namedArguments[0], "'vardef' argument");
          String value = Type.STRING.convert(namedArguments[1], "'vardef' argument");

          String platformSetRegexp = MakeEnvironment.MATCH_ANY;
          if (namedArguments[2] != null) {
            String nickname = Type.STRING.convert(namedArguments[2], "'vardef' argument");
            platformSetRegexp = makeEnv.getPlatformSetRegexp(nickname);
            if (platformSetRegexp == null) {
              throw new EvalException(ast.getLocation(),
                  "invalid platform-set name '" + nickname + "' in call to vardef");
            }
          }
          makeEnv.update(varname, value, platformSetRegexp);
          return 0;
        }
      };
  }

  /**
   * Returns a function which implements "varref". Due to the staged-evaluation
   * nature of google3, this is simply defined as:
   *
   * <pre>
   *     def varref(x):
   *       return '$(%s)' % x
   * </pre>
   *
   * The real work of evaluation happens in the second stage.
   */
  static Function newVarRefFunction() {
    // varref(varname)
    return new PositionalFunction("varref", 1, 1) {
        @Override
        public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
          String varname = Type.STRING.convert(args.get(0), "'varref' argument");
          return "$(" + varname + ")";
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
    setupGlobalEnvironment(env);
    return env;
  }

  /**
   * Set up GLOBAL build environment (c.f. gconfig/guild //tools:GLOBALS).
   */
  private static void setupGlobalEnvironment(Environment env) {
    // varref actually has nothing to do with the MakeEnvironment: it's just a
    // string function, 'x' -> '$(x)'.
    env.update("varref", newVarRefFunction());

    env.update("sysname", "linux");
    env.update("opt_level", "$(BINMODE)");
    env.update("cxx_abi", "$(ABI)");
    env.update("cxx_abi_ref", "$(ABI)");
    env.update("generic_cpu", "$(TARGET_CPU)");
    env.update("glibc_version", "$(GLIBC_VERSION)");
  }

  /****************************************************************************
   * Package creation.
   */

  /**
   * Loads, scans parses and evaluates the build file at "buildFile", and
   * creates and returns a new Package instance with name "packageName". Any
   * errors encountered are reported via "reporter".
   *
   * <p>This method assumes "packageName" is a valid package name according to the
   * {@link LabelValidator#validatePackageName} heuristic.
   *
   * <p>This method allows the caller to inject build file contents by
   * specifying the {@code replacementSource} parameter. If {@code null}, the
   * contents are loaded from the {@code buildFile}.
   *
   * <p>See {@link #evaluateBuildFile} for information on AST retention.
   *
   * @return the newly-created Package instance (which may contain errors)
   */
  public LegacyPackage createPackage(String packageName, Path buildFile,
      CachingPackageLocator locator, ParserInputSource replacementSource,
      RuleVisibility defaultVisibility,
      @Nullable BulkPackageLocatorForCrossingSubpackageBoundaries bulkPackageLocator)
      throws InterruptedException {
    profiler.startTask(ProfilerTask.CREATE_PACKAGE, packageName);
    StoredErrorEventListener localReporter = new StoredErrorEventListener();
    GlobCache globCache = createGlobCache(buildFile.getParentDirectory(), packageName, locator);
    try {
      // Run the lexer and parser with a local reporter, so that errors from other threads do not
      // show up below. Merge the local and global reporters afterwards.
      // Logged message is used as a testability hook tracing the parsing progress
      LOG.fine("Starting to parse " + packageName);

      BuildFileAST buildFileAST;
      boolean hasPreprocessorError = false;
      // TODO(bazel-team): It would be nicer to always pass in the right value rather than rely
      // on the null value.
      Preprocessor.Result preprocessingResult = replacementSource == null
          ? getParserInput(packageName, buildFile, globCache, localReporter)
          : Preprocessor.Result.success(replacementSource, false);
      if (localReporter.hasErrors()) {
        hasPreprocessorError = true;
      }
      buildFileAST = BuildFileAST.parseBuildFile(preprocessingResult.result, localReporter,
                                                 locator, false);
      // Logged message is used as a testability hook tracing the parsing progress
      LOG.fine("Finished parsing of " + packageName);

      MakeEnvironment.Builder makeEnv = new MakeEnvironment.Builder();
      if (platformSetRegexps != null) {
        makeEnv.setPlatformSetRegexps(platformSetRegexps);
      }

      // At this point the package is guaranteed to exist.  It may have parse or
      // evaluation errors, resulting in a diminished number of rules.
      prefetchGlobs(packageName, buildFileAST, preprocessingResult.preprocessed,
          buildFile, globCache, defaultVisibility, makeEnv);

      return evaluateBuildFile(
          packageName, buildFileAST, buildFile, globCache, localReporter.getEvents(),
          defaultVisibility, hasPreprocessorError,
          preprocessingResult.containsTransientErrors, bulkPackageLocator, locator, makeEnv);
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
  public LegacyPackage createPackageForTesting(String packageName, Path buildFile,
      CachingPackageLocator locator, ErrorEventListener listener)
          throws NoSuchPackageException, InterruptedException {
    String error = LabelValidator.validatePackageName(packageName);
    if (error != null) {
      throw new BuildFileNotFoundException(packageName,
          "illegal package name: '" + packageName + "' (" + error + ")");
    }
    LegacyPackage result = createPackage(packageName, buildFile, locator, null,
        ConstantRuleVisibility.PUBLIC, LegacyPackage.EMPTY_BULK_PACKAGE_LOCATOR);
    Event.replayEventsOn(listener, result.getEvents());
    return result;
  }

  /**
   * Returns the parser input (with preprocessing already applied, if
   * applicable) for the specified package and build file.
   *
   * @param packageName the name of the package; used for error messages
   * @param buildFile the path of the BUILD file to read
   * @param locator package locator used in recursive globbing
   * @param listener the listener on which preprocessing errors/warnings are to
   *        be reported
   * @throws NoSuchPackageException if the build file cannot be read
   * @return the preprocessed input, as seen by Blaze's parser
   */
  // Used externally!
  public ParserInputSource getParserInput(String packageName, Path buildFile,
      CachingPackageLocator locator, ErrorEventListener listener)
      throws NoSuchPackageException, InterruptedException {
    return getParserInput(
        packageName, buildFile,
        createGlobCache(buildFile.getParentDirectory(), packageName, locator),
        listener).result;
  }

  private GlobCache createGlobCache(Path packageDirectory, String packageName,
      CachingPackageLocator locator) {
    return new GlobCache(packageDirectory, packageName, locator, syscalls, threadPool);
  }

  /**
   * Version of #getParserInput(String, Path, GlobCache, Reporter) that allows
   * to inject a glob cache that gets populated during preprocessing.
   */
  private Preprocessor.Result getParserInput(
      String packageName, Path buildFile, GlobCache globCache, ErrorEventListener listener)
          throws InterruptedException {
    ParserInputSource inputSource;
    try {
      inputSource = ParserInputSource.create(buildFile);
    } catch (IOException e) {
      listener.error(Location.fromFile(buildFile), e.getMessage());
      return Preprocessor.Result.transientError(buildFile);
    }

    if (preprocessor == null) {
      return Preprocessor.Result.success(inputSource, false);
    }

    try {
      return preprocessor.preprocess(inputSource, packageName, globCache, listener,
                                            globalEnv, ruleFactory.getRuleClassNames());
    } catch (IOException e) {
      listener.error(Location.fromFile(buildFile),
                     "preprocessing failed: " + e.getMessage());
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
  private static class PackageContext {

    final LegacyPackage.LegacyPackageBuilder pkgBuilder;
    final ErrorEventListener listener;
    final boolean retainASTs;

    PackageContext(LegacyPackage.LegacyPackageBuilder pkgBuilder, ErrorEventListener listener,
        boolean retainASTs) {
      this.pkgBuilder = pkgBuilder;
      this.listener = listener;
      this.retainASTs = retainASTs;
    }
  }

  private static void buildPkgEnv(Environment pkgEnv, String packageName,
      MakeEnvironment.Builder pkgMakeEnv, PackageContext context) {
    pkgEnv.update("distribs", newDistribsFunction(context));
    pkgEnv.update("glob", newGlobFunction(context, /*async=*/false));
    pkgEnv.update("select", newSelectFunction());
    pkgEnv.update("FilesetEntry", newFilesetEntryFunction(context));
    pkgEnv.update("mocksubinclude", newMockSubincludeFunction(context));
    pkgEnv.update("licenses", newLicensesFunction(context));
    pkgEnv.update("vardef", newVarDefFunction(pkgMakeEnv));
    pkgEnv.update("exports_files", newExportsFilesFunction(context));
    pkgEnv.update("package_group", newPackageGroupFunction(context));
    pkgEnv.update("package", newPackageFunction(context));
    pkgEnv.update("subinclude", newSubincludeFunction());

    pkgEnv.update("PACKAGE_NAME", packageName);
  }

  private void buildPkgEnv(Environment pkgEnv, String packageName,
      MakeEnvironment.Builder pkgMakeEnv, PackageContext context, RuleFactory ruleFactory,
      SkylarkRuleFactory skylarkRuleFactory) {
    pkgEnv.setImportAllowed(skylarkEnabled);
    buildPkgEnv(pkgEnv, packageName, pkgMakeEnv, context);
    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      pkgEnv.update(ruleClass,
          newRuleFunction(ruleFactory, skylarkRuleFactory, ruleClass, null, context));
    }
  }

  private Environment loadSkylarkExtension(Path file, CachingPackageLocator locator,
      Path root, PackageContext context, ImmutableList<Path> extensionFileStack)
          throws InterruptedException {
    BuildFileAST buildFileAST;
    try {
      buildFileAST = BuildFileAST.parseSkylarkFile(file, context.listener, locator,
          ruleClassProvider.getSkylarkValidationEnvironment());
    } catch (IOException e) {
      context.listener.error(Location.fromFile(file), e.getMessage());
      return null;
    }

    Environment env = skylarkRuleFactory.getSkylarkRuleClassEnvironment(file);
    env.setImportAllowed(true);

    if (!loadAllImports(
        buildFileAST, root, file, locator, env, context, extensionFileStack, false)) {
      return null;
    }

    if (!buildFileAST.exec(env, context.listener)) {
      return null;
    }

    return env;
  }

  // Load all extensions imported in buildFileAST and update the environment.
  private boolean loadAllImports(BuildFileAST buildFileAST, Path root, Path parentFile,
      CachingPackageLocator locator, Environment parentEnv, PackageContext context,
      ImmutableList<Path> extensionFileStack, boolean updateSkylarkRuleFactory)
      throws InterruptedException {
    // TODO(bazel-team): We should have a global cache and make sure each
    // imported file is loaded at most once.
    Map<PathFragment, Environment> imports = new HashMap<>();
    for (PathFragment imp : buildFileAST.getImports()) {
      Path file = root.getRelative(imp);
      if (extensionFileStack.contains(file)) {
        context.listener.error(Location.fromFile(parentFile), "Recursive import: " + file);
        parentEnv.setImportedExtensions(imports);
        return false;
      }
      // TODO(bazel-team): Update the Skyframe code path in PackageNodeBuilder too.
      if (updateSkylarkRuleFactory) {
        skylarkRuleFactory.clear(file);
      }
      Environment extensionEnv = loadSkylarkExtension(file, locator, root, context,
          ImmutableList.<Path>builder().addAll(extensionFileStack).add(file).build());
      if (extensionEnv == null) {
        parentEnv.setImportedExtensions(imports);
        return false;
      }
      if (updateSkylarkRuleFactory) {
        for (String ruleClass : skylarkRuleFactory.getRuleClassNames(file)) {
          extensionEnv.update(ruleClass,
              newRuleFunction(ruleFactory, skylarkRuleFactory, ruleClass, file, context));
        }
      }
      imports.put(imp, extensionEnv);
    }
    parentEnv.setImportedExtensions(imports);
    return true;
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
  public LegacyPackage evaluateBuildFile(String packageName, BuildFileAST buildFileAST,
      Path buildFilePath, GlobCache globCache, Iterable<Event> pastEvents,
      RuleVisibility defaultVisibility, boolean containsError, boolean containsTransientError,
      @Nullable BulkPackageLocatorForCrossingSubpackageBoundaries bulkPackageLocator,
      CachingPackageLocator locator,
      MakeEnvironment.Builder pkgMakeEnv)
      throws InterruptedException {
    // Important: Environment should be unreachable by the end of this method!
    Environment pkgEnv = new Environment(globalEnv);

    LegacyPackage.LegacyPackageBuilder pkgBuilder =
        new LegacyPackage.LegacyPackageBuilder(packageName)
        .setFilename(buildFilePath)
        .setAST(retainAsts ? buildFileAST : null)
        .setMakeEnv(pkgMakeEnv)
        .setGlobCache(globCache)
        .setDefaultVisibility(defaultVisibility)
        // "defaultVisibility" comes from the command line. Let's give the BUILD file a chance to
        // set default_visibility once, be reseting the PackageBuilder.defaultVisibilitySet flag.
        .setDefaultVisibilitySet(false);

    StoredErrorEventListener listener = new StoredErrorEventListener();
    Event.replayEventsOn(listener, pastEvents);

    // Stuff that closes over the package context:
    PackageContext context = new PackageContext(pkgBuilder, listener, retainAsts);
    buildPkgEnv(pkgEnv, packageName, pkgMakeEnv, context, ruleFactory, skylarkRuleFactory);

    if (containsError) {
      pkgBuilder.setContainsErrors();
    }

    if (containsTransientError) {
      pkgBuilder.setContainsTemporaryErrors();
    }

    if (!validatePackageName(packageName, buildFileAST.getLocation(), listener)) {
      pkgBuilder.setContainsErrors();
    }

    Path root = Package.getSourceRoot(buildFilePath, new PathFragment(packageName));
    if (!loadAllImports(buildFileAST, root, buildFilePath, locator, pkgEnv,
        context, ImmutableList.<Path>of(buildFilePath), true)) {
      pkgBuilder.setContainsErrors();
    }

    if (!validateAssignmentStatements(pkgEnv, buildFileAST, listener)) {
      pkgBuilder.setContainsErrors();
    }

    if (buildFileAST.containsErrors()) {
      pkgBuilder.setContainsErrors();
    }

    // TODO(bazel-team): (2009) the invariant "if errors are reported, mark the package
    // as containing errors" is strewn all over this class.  Refactor to use an
    // event sensor--and see if we can simplify the calling code in
    // createPackage().
    if (!buildFileAST.exec(pkgEnv, listener)) {
      pkgBuilder.setContainsErrors();
    }

    return pkgBuilder.build(bulkPackageLocator, listener);
  }

  /**
   * Visit all targets and expand the globs in parallel.
   */
  private void prefetchGlobs(String packageName, BuildFileAST buildFileAST,
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

    LegacyPackage.LegacyPackageBuilder pkgBuilder =
        new LegacyPackage.LegacyPackageBuilder(packageName)
        .setFilename(buildFilePath)
        .setMakeEnv(pkgMakeEnv)
        .setGlobCache(globCache)
        .setDefaultVisibility(defaultVisibility)
        // "defaultVisibility" comes from the command line. Let's give the BUILD file a chance to
        // set default_visibility once, be reseting the PackageBuilder.defaultVisibilitySet flag.
        .setDefaultVisibilitySet(false);

    // Stuff that closes over the package context:
    PackageContext context = new PackageContext(pkgBuilder, NullErrorEventListener.INSTANCE, false);
    buildPkgEnv(pkgEnv, packageName, pkgMakeEnv, context);
    pkgEnv.update("glob", newGlobFunction(context, /*async=*/true));
    // The Fileset function is heavyweight in that it can run glob(). Avoid this during the
    // preloading phase.
    pkgEnv.remove("FilesetEntry");

    buildFileAST.exec(pkgEnv, NullErrorEventListener.INSTANCE);
  }


  /**
   * Tests a build AST to ensure that it contains no assignment statements that
   * redefine built-in build rules.
   *
   * @param pkgEnv a package environment initialized with all of the built-in
   *        build rules
   * @param ast the build file AST to be tested
   * @param listener a listener where any errors should be logged
   * @return true if the build file contains no redefinitions of built-in
   *         functions
   */
  private static boolean validateAssignmentStatements(Environment pkgEnv,
                                                      BuildFileAST ast,
                                                      ErrorEventListener listener) {
    for (Statement stmt : ast.getStatements()) {
      if (stmt instanceof AssignmentStatement) {
        Expression lvalue = ((AssignmentStatement) stmt).getLValue();
        if (!(lvalue instanceof Ident)) {
          continue;
        }
        String target = ((Ident) lvalue).getName();
        if (pkgEnv.lookup(target, null) != null) {
          listener.error(stmt.getLocation(), "Reassignment of builtin build "
              + "function '" + target + "' not permitted");
          return false;
        }
      }
    }
    return true;
  }

  // Reports an error and returns false iff package name was illegal.
  private static boolean validatePackageName(String name, Location location,
      ErrorEventListener listener) {
    String error = LabelValidator.validatePackageName(name);
    if (error != null) {
      listener.error(location, error);
      return false; // Invalid package name 'foo'
    }
    return true;
  }
}
