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

package com.google.devtools.build.lib.skyframe;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.BazelCompileContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.BazelStarlarkEnvironment;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.BzlCompileValue.TypeOptions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.AssignmentStatement;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.Identifier;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.Statement;
import net.starlark.java.syntax.StringLiteral;
import net.starlark.java.syntax.SyntaxError;

/**
 * A Skyframe function that compiles the .bzl file denoted by a Label.
 *
 * <p>Given a {@link Label} referencing a Starlark file, BzlCompileFunction loads, parses, resolves,
 * and compiles it. The Label must be absolute, and must not reference the special {@code external}
 * package. If the file (or the package containing it) doesn't exist, the function doesn't fail, but
 * instead returns a specific {@code NO_FILE} {@link BzlCompileValue}.
 */
// TODO(adonovan): actually compile. The name is a step ahead of the implementation.
public class BzlCompileFunction implements SkyFunction {

  private final BazelStarlarkEnvironment bazelStarlarkEnvironment;
  private final HashFunction hashFunction;
  private final PackageLoadingListener packageLoadingListener;

  public BzlCompileFunction(
      BazelStarlarkEnvironment bazelStarlarkEnvironment,
      HashFunction hashFunction,
      PackageLoadingListener packageLoadingListener) {
    this.bazelStarlarkEnvironment = bazelStarlarkEnvironment;
    this.hashFunction = hashFunction;
    this.packageLoadingListener = packageLoadingListener;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    try {
      return computeInline(
          (BzlCompileValue.Key) skyKey.argument(),
          env,
          bazelStarlarkEnvironment,
          hashFunction,
          packageLoadingListener);
    } catch (FailedIOException e) {
      throw new FunctionException(e);
    }
  }

  @Nullable
  static BzlCompileValue computeInline(
      BzlCompileValue.Key key,
      Environment env,
      BazelStarlarkEnvironment bazelStarlarkEnvironment,
      HashFunction hashFunction,
      PackageLoadingListener packageLoadingListener)
      throws FailedIOException, InterruptedException {
    byte[] bytes;
    byte[] digest;
    String inputName;
    RootedPath rootedPath = null;

    StarlarkSemantics semantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (semantics == null) {
      return null;
    }

    if (key.kind == BzlCompileValue.Kind.EMPTY_PRELUDE) {
      // Default prelude is empty.
      bytes = new byte[] {};
      digest = null;
      inputName = "<default prelude>";
    } else {
      // Obtain the file.
      rootedPath = RootedPath.toRootedPath(key.root, key.label.toPathFragment());
      SkyKey fileSkyKey = FileValue.key(rootedPath);
      FileValue fileValue = null;
      try {
        fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);
      } catch (IOException e) {
        throw new FailedIOException(e, Transience.PERSISTENT);
      }
      if (fileValue == null) {
        return null;
      }

      if (fileValue.exists()) {
        if (!fileValue.isFile()) {
          return fileValue.isDirectory()
              ? BzlCompileValue.noFile("cannot load '%s': is a directory", key.label)
              : BzlCompileValue.noFile(
                  "cannot load '%s': not a regular file (dangling link?)", key.label);
        }

        Path path = rootedPath.asPath();
        Location location = Location.fromFile(path.toString());

        long fileSize = fileValue.getSize();
        // Special files (such as named pipes or sockets used to coordinate timing in tests) do not
        // report a meaningful size in stat(), so their size is checked after reading below.
        if (key.kind == BzlCompileValue.Kind.NORMAL && !fileValue.isSpecialFile()) {
          BzlCompileValue limitFailure =
              checkHardFileSizeLimit(
                  key, fileSize, location, semantics, bazelStarlarkEnvironment, env);
          if (env.valuesMissing()) {
            return null;
          }
          if (limitFailure != null) {
            return limitFailure;
          }
        }

        // Read the file.
        try {
          bytes =
              fileValue.isSpecialFile()
                  ? FileSystemUtils.readContent(path)
                  : FileSystemUtils.readWithKnownFileSize(path, fileSize);
        } catch (IOException e) {
          throw new FailedIOException(e, Transience.TRANSIENT);
        }

        if (key.kind == BzlCompileValue.Kind.NORMAL && fileValue.isSpecialFile()) {
          BzlCompileValue limitFailure =
              checkHardFileSizeLimit(
                  key, bytes.length, location, semantics, bazelStarlarkEnvironment, env);
          if (env.valuesMissing()) {
            return null;
          }
          if (limitFailure != null) {
            return limitFailure;
          }
        }

        digest = fileValue.getDigest(); // may be null
        inputName = path.toString();
      } else {
        if (key.kind == BzlCompileValue.Kind.PRELUDE) {
          // A non-existent prelude is fine.
          bytes = new byte[] {};
          digest = null;
          inputName = "<default prelude>";
        } else {
          return BzlCompileValue.noFile("cannot load '%s': no such file", key.label);
        }
      }
    }

    // Compute digest if we didn't already get it from a fileValue.
    if (digest == null) {
      digest = hashFunction.hashBytes(bytes).asBytes();
    }

    TypeOptions typeOptions = getTypeOptions(semantics, key);
    boolean resolveTypeSyntax =
        typeOptions.wantStaticTypeChecking() || typeOptions.wantDynamicTypeChecking();

    ImmutableMap<String, Object> predeclared;
    if (key.isSclDialect()) {
      predeclared = bazelStarlarkEnvironment.getStarlarkGlobals().getSclToplevels();
    } else if (key.kind == BzlCompileValue.Kind.BUILTINS) {
      predeclared =
          resolveTypeSyntax
              ? bazelStarlarkEnvironment.getBuiltinsBzlEnvWithExtraTypeConstructors()
              : bazelStarlarkEnvironment.getBuiltinsBzlEnv();
    } else {
      // Use the predeclared environment for BUILD-loaded bzl files, ignoring injection. It is not
      // the right env for the actual evaluation of BUILD-loaded bzl files because it doesn't
      // map to the injected symbols. But the names of the symbols are the same, and the names are
      // all we need to do symbol resolution.
      //
      // For WORKSPACE-loaded bzl files, the env isn't quite right not because of injection but
      // because the "native" object is different. But A) that will be fixed with #11954, and B) we
      // don't care for the same reason as above.

      predeclared =
          resolveTypeSyntax
              ? bazelStarlarkEnvironment.getUninjectedBuildBzlEnvWithExtraTypeConstructors()
              : bazelStarlarkEnvironment.getUninjectedBuildBzlEnv();
    }

    // We have all deps. Parse, resolve, and return.
    ParserInput input;
    try {
      input =
          StarlarkUtil.createParserInput(
              bytes,
              inputName,
              semantics.get(BuildLanguageOptions.INCOMPATIBLE_ENFORCE_STARLARK_UTF8),
              env.getListener());
    } catch (
        @SuppressWarnings("UnusedException") // createParserInput() reports its own error message
        StarlarkUtil.InvalidUtf8Exception e) {
      return BzlCompileValue.noFile("compilation of '%s' failed", inputName);
    }

    FileOptions.Builder optionsBuilder =
        FileOptions.builder()
            // By default, Starlark load statements create file-local bindings.
            // However, the BUILD prelude typically contains nothing but load
            // statements whose bindings are intended to be visible in all BUILD
            // files. The loadBindsGlobally flag allows us to retrieve them.
            .loadBindsGlobally(key.isBuildPrelude())
            // .scl files should be ASCII-only in string literals.
            // TODO(bazel-team): It'd be nice if we could intercept non-ASCII errors from the lexer,
            // and modify the displayed message to clarify to the user that the string would be
            // permitted in a .bzl file. But there's no easy way to do that short of either string
            // matching the error message or reworking the interpreter API to put more structured
            // detail in errors (i.e. new fields or error subclasses).
            .stringLiteralsAreAsciiOnly(key.isSclDialect());
    updateFileOptions(optionsBuilder, typeOptions);
    StarlarkFile file = StarlarkFile.parse(input, optionsBuilder.build());

    if (key.kind == BzlCompileValue.Kind.NORMAL) {
      BzlCompileValue softLimitFailure =
          checkSoftFileSizeLimit(
              key,
              bytes.length,
              file,
              Location.fromFile(inputName),
              semantics,
              bazelStarlarkEnvironment,
              env);
      if (env.valuesMissing()) {
        return null;
      }
      if (softLimitFailure != null) {
        return softLimitFailure;
      }
    }

    // compile
    final Module module;

    if (key.kind == BzlCompileValue.Kind.EMPTY_PRELUDE) {
      // The empty prelude has no label, so we can't use it to filter the predeclareds.
      // This doesn't matter since the empty prelude doesn't attempt to access any predeclareds
      // anyway.
      module = Module.withPredeclared(semantics, predeclared);
    } else {
      // The BazelCompileContext holds additional contextual info to be associated with the Module
      // The information is used to filter predeclareds
      BazelCompileContext bazelCompileContext =
          BazelCompileContext.create(key.label, file.getName());
      module = Module.withPredeclaredAndData(semantics, predeclared, bazelCompileContext);
    }
    try {
      Program prog = Program.compileFile(file, module);
      if (key.kind == BzlCompileValue.Kind.NORMAL) {
        packageLoadingListener.onBzlCompileCompleteAndSuccessful(rootedPath, bytes.length);
      }
      return BzlCompileValue.withProgram(prog, digest, typeOptions);
    } catch (SyntaxError.Exception ex) {
      addSyntaxErrorsToListener(env.getListener(), ex.errors(), key);
      return BzlCompileValue.noFile(
          "compilation of module '%s'%s failed",
          key.label.toPathFragment(),
          StarlarkBuiltinsValue.isBuiltinsRepo(key.label.getRepository()) ? " (internal)" : "");
    }
  }

  /**
   * Whether the file should permit type syntax (annotations, etc.) based on flags and the type of
   * file.
   */
  private static TypeOptions getTypeOptions(StarlarkSemantics semantics, BzlCompileValue.Key key) {
    boolean typeSyntaxFlag =
        semantics.getBool(BuildLanguageOptions.EXPERIMENTAL_STARLARK_TYPE_SYNTAX);
    List<String> allowlist =
        semantics.get(BuildLanguageOptions.EXPERIMENTAL_STARLARK_TYPES_ALLOWED_PATHS);

    boolean okFiletype =
        // annotations in prelude not allowed (it has null key.label)
        !key.isBuildPrelude()
            // annotations in SCL not allowed (not yet compatible with Go-Starlark interpreter)
            && !key.isSclDialect();

    boolean useTypeSyntax = false;
    if (okFiletype) {
      if (typeSyntaxFlag) {
        if (allowlist.isEmpty()
            || allowlist.stream().anyMatch(s -> key.label.getCanonicalForm().startsWith(s))) {
          useTypeSyntax = true;
        }
      }
      if (key.isBuiltins()) {
        // Always enable type syntax for @_builtins
        useTypeSyntax = true;
      }
    }
    boolean doStaticTypeChecking =
        useTypeSyntax
            && (semantics.getBool(StarlarkSemantics.EXPERIMENTAL_STARLARK_STATIC_TYPE_CHECKING)
                // Always enable static type checking for @_builtins
                || key.isBuiltins());
    boolean doDynamicTypeChecking =
        useTypeSyntax
            && semantics.getBool(StarlarkSemantics.EXPERIMENTAL_STARLARK_DYNAMIC_TYPE_CHECKING);

    return new TypeOptions(useTypeSyntax, doStaticTypeChecking, doDynamicTypeChecking);
  }

  private static void updateFileOptions(FileOptions.Builder builder, TypeOptions typeOptions) {
    boolean needsTypeInfo =
        typeOptions.wantStaticTypeChecking() || typeOptions.wantDynamicTypeChecking();
    builder
        .allowTypeSyntax(typeOptions.useTypeSyntax())
        .resolveTypeSyntax(needsTypeInfo)
        .tolerateInvalidTypeExpressions(!needsTypeInfo);
  }

  /**
   * Replays the syntax errors from a file onto an event handler, adding more context if necessary.
   */
  private static void addSyntaxErrorsToListener(
      EventHandler handler, List<SyntaxError> errors, BzlCompileValue.Key key) {
    Event.replayEventsOn(handler, errors);
    // If type annotations are disallowed, it could either be because the required flags aren't
    // enabled or because the filetype disallows it.
    for (var err : errors) {
      if (err.message().contains(": type annotations are disallowed")) {
        Location fileLoc = Location.fromFile(err.location().file());
        String explanation =
            key.isSclDialect()
                ? "Type annotations are not permitted in .scl files."
                : """
                Type annotations syntax can be enabled with --experimental_starlark_type_syntax \
                and/or --experimental_starlark_types_allowed_paths.\
                """;
        handler.handle(Event.error(fileLoc, explanation));
      }
    }
  }

  static final class FailedIOException extends Exception {
    private final Transience transience;

    FailedIOException(IOException cause, Transience transience) {
      super(cause.getMessage(), cause);
      this.transience = transience;
    }

    Transience getTransience() {
      return transience;
    }
  }

  private static final class FunctionException extends SkyFunctionException {
    private FunctionException(FailedIOException cause) {
      super(cause, cause.transience);
    }
  }

  private static final String KNOWN_OVERSIZED_BZL_FILE_SYMBOL = "_KNOWN_OVERSIZED_BZL_FILE";

  @Nullable
  private static Label parseAllowlistLabel(String raw) {
    if (raw.isEmpty()) {
      return null;
    }
    return Label.parseCanonicalUnchecked(raw);
  }

  private record AllowlistCheckResult(boolean allowlisted, @Nullable String errorMessage) {
    static final AllowlistCheckResult ALLOWLISTED = new AllowlistCheckResult(true, null);
    static final AllowlistCheckResult NOT_ALLOWLISTED = new AllowlistCheckResult(false, null);

    static AllowlistCheckResult error(String errorMessage) {
      return new AllowlistCheckResult(false, errorMessage);
    }
  }

  private static final Cache<BzlCompileValue, ParsedAllowlist> allowlistCache =
      CacheBuilder.newBuilder().weakKeys().build();

  private record ParsedAllowlist(
      @Nullable ImmutableSet<String> allowedEntries, @Nullable String errorMessage) {
    static ParsedAllowlist success(ImmutableSet<String> allowedEntries) {
      return new ParsedAllowlist(allowedEntries, null);
    }

    static ParsedAllowlist error(String errorMessage) {
      return new ParsedAllowlist(null, errorMessage);
    }
  }

  private static ParsedAllowlist evaluateAllowlist(
      Label allowlistLabel,
      BzlCompileValue allowlistCompileValue,
      StarlarkSemantics semantics,
      BazelStarlarkEnvironment bazelStarlarkEnvironment)
      throws InterruptedException {
    if (!allowlistCompileValue.lookupSuccessful()) {
      return ParsedAllowlist.error(
          String.format(
              "Failed to compile allowlist file '%s': %s",
              allowlistLabel.getCanonicalForm(), allowlistCompileValue.getError()));
    }

    ImmutableMap<String, Object> predeclared =
        bazelStarlarkEnvironment.getStarlarkGlobals().getSclToplevels();
    Module module = Module.withPredeclared(semantics, predeclared);
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, semantics);
      Starlark.execFileProgram(allowlistCompileValue.getProgram(), module, thread);
    } catch (EvalException e) {
      return ParsedAllowlist.error(
          String.format(
              "Failed to evaluate allowlist file '%s': %s",
              allowlistLabel.getCanonicalForm(), e.getMessage()));
    }

    Object allowedObj = module.getGlobal("ALLOWED");
    if (allowedObj == null) {
      return ParsedAllowlist.error(
          String.format(
              "Allowlist file '%s' does not define an 'ALLOWED' symbol.",
              allowlistLabel.getCanonicalForm()));
    }
    if (!(allowedObj instanceof Sequence<?> allowedSeq)) {
      return ParsedAllowlist.error(
          String.format(
              "Allowlist file '%s': 'ALLOWED' must be a list of strings, but got %s.",
              allowlistLabel.getCanonicalForm(), Starlark.type(allowedObj)));
    }
    ImmutableSet.Builder<String> allowedBuilder = ImmutableSet.builder();
    for (Object item : allowedSeq) {
      if (item instanceof String s) {
        allowedBuilder.add(s);
      }
    }
    return ParsedAllowlist.success(allowedBuilder.build());
  }

  @Nullable
  private static BzlCompileValue checkHardFileSizeLimit(
      BzlCompileValue.Key key,
      long fileSize,
      Location location,
      StarlarkSemantics semantics,
      BazelStarlarkEnvironment bazelStarlarkEnvironment,
      Environment env)
      throws InterruptedException {
    long maxLimit = semantics.get(BuildLanguageOptions.MAX_BZL_FILE_SIZE);
    if (maxLimit <= 0 || fileSize <= maxLimit) {
      return null;
    }
    AllowlistCheckResult allowlistResult =
        isFileAllowlisted(key, fileSize, maxLimit, semantics, bazelStarlarkEnvironment, env);
    if (allowlistResult == null || allowlistResult.allowlisted()) {
      return null;
    }
    String msg =
        allowlistResult.errorMessage() != null
            ? allowlistResult.errorMessage()
            : String.format(
                "File '%s' size (%d bytes) exceeds the maximum allowed size (%d bytes). Loading"
                    + " oversized Starlark files is blocked to protect Blaze from JVM memory"
                    + " exhaustion.",
                key.label, fileSize, maxLimit);
    env.getListener().handle(Event.error(location, msg));
    return BzlCompileValue.noFile("%s", msg);
  }

  @Nullable
  private static BzlCompileValue checkSoftFileSizeLimit(
      BzlCompileValue.Key key,
      long fileSize,
      StarlarkFile file,
      Location location,
      StarlarkSemantics semantics,
      BazelStarlarkEnvironment bazelStarlarkEnvironment,
      Environment env)
      throws InterruptedException {
    long softLimit = semantics.get(BuildLanguageOptions.SOFT_MAX_BZL_FILE_SIZE);
    if (softLimit <= 0 || fileSize <= softLimit || hasKnownOversizedOptOut(file)) {
      return null;
    }
    AllowlistCheckResult allowlistResult =
        isFileAllowlisted(key, fileSize, softLimit, semantics, bazelStarlarkEnvironment, env);
    if (allowlistResult == null || allowlistResult.allowlisted()) {
      return null;
    }
    String msg =
        allowlistResult.errorMessage() != null
            ? allowlistResult.errorMessage()
            : String.format(
                "File '%s' size (%d bytes) exceeds the soft maximum size (%d bytes). Split this"
                    + " file or move data out of Starlark. To opt out up to the hard limit, set"
                    + " `%s = \"%s\"` at the top level of the file.",
                key.label,
                fileSize,
                softLimit,
                KNOWN_OVERSIZED_BZL_FILE_SYMBOL,
                BuildLanguageOptions.KNOWN_OVERSIZED_BZL_FILE_VALUE_EXAMPLE);
    env.getListener().handle(Event.error(location, msg));
    return BzlCompileValue.noFile("%s", msg);
  }

  static boolean registerAllowlistDepIfOversized(
      BzlCompileValue.Key key, SkyValue bzlFileSkyValue, Environment env)
      throws InterruptedException {
    if (!(bzlFileSkyValue instanceof FileValue bzlFileValue)
        || key.kind != BzlCompileValue.Kind.NORMAL
        || !bzlFileValue.exists()
        || !bzlFileValue.isFile()) {
      return true;
    }
    StarlarkSemantics semantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (semantics == null) {
      return false;
    }
    Label allowlistLabel =
        parseAllowlistLabel(semantics.get(BuildLanguageOptions.BZL_FILE_SIZE_LIMIT_ALLOWLIST));
    if (allowlistLabel == null || key.label == null || key.label.equals(allowlistLabel)) {
      return true;
    }
    long maxLimit = semantics.get(BuildLanguageOptions.MAX_BZL_FILE_SIZE);
    long softLimit = semantics.get(BuildLanguageOptions.SOFT_MAX_BZL_FILE_SIZE);
    boolean exceedsLimit =
        bzlFileValue.isSpecialFile()
            || (maxLimit > 0 && bzlFileValue.getSize() > maxLimit)
            || (softLimit > 0 && bzlFileValue.getSize() > softLimit);
    if (!exceedsLimit) {
      return true;
    }

    PathFragment dir = Label.getContainingDirectory(allowlistLabel);
    PackageIdentifier dirId = PackageIdentifier.create(allowlistLabel.getRepository(), dir);
    ContainingPackageLookupValue packageLookup;
    try {
      packageLookup =
          (ContainingPackageLookupValue)
              env.getValueOrThrow(
                  ContainingPackageLookupValue.key(dirId),
                  BuildFileNotFoundException.class,
                  InconsistentFilesystemException.class);
    } catch (BuildFileNotFoundException | InconsistentFilesystemException e) {
      return true;
    }
    if (packageLookup == null || !packageLookup.hasContainingPackage()) {
      return !env.valuesMissing();
    }
    Root root = packageLookup.getContainingPackageRoot();

    SkyKey allowlistFileKey =
        FileValue.key(RootedPath.toRootedPath(root, allowlistLabel.toPathFragment()));
    try {
      var unused = env.getValueOrThrow(allowlistFileKey, IOException.class);
    } catch (IOException e) {
      return true;
    }
    if (env.valuesMissing()) {
      return false;
    }

    BzlCompileValue.Key allowlistCompileKey = BzlCompileValue.key(root, allowlistLabel);
    try {
      var unused = env.getValueOrThrow(allowlistCompileKey, FailedIOException.class);
    } catch (FailedIOException e) {
      return true;
    }
    return !env.valuesMissing();
  }

  private static boolean hasKnownOversizedOptOut(StarlarkFile file) {
    for (Statement stmt : file.getStatements()) {
      if (stmt instanceof AssignmentStatement assignment
          && assignment.getOperator() == null
          && assignment.getLHS() instanceof Identifier id
          && id.getName().equals(KNOWN_OVERSIZED_BZL_FILE_SYMBOL)
          && assignment.getRHS() instanceof StringLiteral str
          && BuildLanguageOptions.KNOWN_OVERSIZED_BZL_FILE_VALUE_PATTERN
              .matcher(str.getValue())
              .matches()) {
        return true;
      }
    }
    return false;
  }

  @Nullable
  private static AllowlistCheckResult isFileAllowlisted(
      BzlCompileValue.Key key,
      long fileSize,
      long limit,
      StarlarkSemantics semantics,
      BazelStarlarkEnvironment bazelStarlarkEnvironment,
      Environment env)
      throws InterruptedException {
    Label allowlistLabel =
        parseAllowlistLabel(semantics.get(BuildLanguageOptions.BZL_FILE_SIZE_LIMIT_ALLOWLIST));
    if (allowlistLabel == null || key.label == null) {
      return AllowlistCheckResult.NOT_ALLOWLISTED;
    }
    if (allowlistLabel.equals(key.label)) {
      return AllowlistCheckResult.error(
          String.format(
              "Allowlist file '%s' size (%d bytes) exceeds the maximum allowed size (%d bytes)."
                  + " The allowlist file itself cannot be oversized to avoid dependency cycles.",
              key.label, fileSize, limit));
    }

    PathFragment dir = Label.getContainingDirectory(allowlistLabel);
    PackageIdentifier dirId = PackageIdentifier.create(allowlistLabel.getRepository(), dir);
    ContainingPackageLookupValue packageLookup;
    try {
      packageLookup =
          (ContainingPackageLookupValue)
              env.getValueOrThrow(
                  ContainingPackageLookupValue.key(dirId),
                  BuildFileNotFoundException.class,
                  InconsistentFilesystemException.class);
    } catch (BuildFileNotFoundException | InconsistentFilesystemException e) {
      return AllowlistCheckResult.error(
          String.format(
              "Failed to find containing package for allowlist '%s': %s",
              allowlistLabel.getCanonicalForm(), e.getMessage()));
    }
    if (packageLookup == null) {
      return null;
    }
    if (!packageLookup.hasContainingPackage()) {
      return AllowlistCheckResult.error(
          String.format(
              "Failed to load allowlist file '%s': no containing package",
              allowlistLabel.getCanonicalForm()));
    }
    Root root = packageLookup.getContainingPackageRoot();

    SkyKey allowlistFileKey =
        FileValue.key(RootedPath.toRootedPath(root, allowlistLabel.toPathFragment()));
    FileValue allowlistFileValue;
    try {
      allowlistFileValue = (FileValue) env.getValueOrThrow(allowlistFileKey, IOException.class);
    } catch (IOException e) {
      return AllowlistCheckResult.error(
          String.format(
              "Failed to read allowlist file '%s': %s",
              allowlistLabel.getCanonicalForm(), e.getMessage()));
    }
    if (allowlistFileValue == null) {
      return null;
    }
    if (!allowlistFileValue.exists()) {
      return AllowlistCheckResult.error(
          String.format("Allowlist file '%s' does not exist.", allowlistLabel.getCanonicalForm()));
    }

    BzlCompileValue.Key allowlistCompileKey = BzlCompileValue.key(root, allowlistLabel);
    BzlCompileValue allowlistCompileValue;
    try {
      allowlistCompileValue =
          (BzlCompileValue) env.getValueOrThrow(allowlistCompileKey, FailedIOException.class);
    } catch (FailedIOException e) {
      return AllowlistCheckResult.error(
          String.format(
              "Failed to compile allowlist file '%s': %s",
              allowlistLabel.getCanonicalForm(), e.getMessage()));
    }
    if (allowlistCompileValue == null) {
      return null;
    }

    ParsedAllowlist parsedAllowlist;
    try {
      parsedAllowlist =
          allowlistCache.get(
              allowlistCompileValue,
              () ->
                  evaluateAllowlist(
                      allowlistLabel, allowlistCompileValue, semantics, bazelStarlarkEnvironment));
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof InterruptedException interruptedException) {
        throw interruptedException;
      }
      return AllowlistCheckResult.error(
          String.format(
              "Failed to evaluate allowlist file '%s': %s",
              allowlistLabel.getCanonicalForm(),
              cause != null ? cause.getMessage() : e.getMessage()));
    }

    if (parsedAllowlist.errorMessage() != null) {
      return AllowlistCheckResult.error(parsedAllowlist.errorMessage());
    }
    ImmutableSet<String> allowed = parsedAllowlist.allowedEntries();
    String pathString = key.label.toPathFragment().getPathString();
    String canonicalLabel = key.label.getCanonicalForm();
    String repoRelativeLabel = "//" + key.label.getPackageName() + ":" + key.label.getName();
    if (allowed != null
        && (allowed.contains(pathString)
            || allowed.contains(canonicalLabel)
            || allowed.contains(repoRelativeLabel))) {
      return AllowlistCheckResult.ALLOWLISTED;
    }
    return AllowlistCheckResult.NOT_ALLOWLISTED;
  }
}
