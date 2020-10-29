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

import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.ATTRIBUTES;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.NATIVE;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.REPOSITORIES;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.RULE_CLASS;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFactory;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.packages.WorkspaceFileValue.WorkspaceFileKey;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.ResolvedFileValue;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.LoadStatement;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.Statement;

/**
 * A SkyFunction to read, parse, and resolve a WORKSPACE file, divide it into chunks, then execute a
 * single chunk. (The read/parse/resolve work is done repeatedly.)
 */
public class WorkspaceFileFunction implements SkyFunction {

  private final PackageFactory packageFactory;
  private final BlazeDirectories directories;
  private final RuleClassProvider ruleClassProvider;
  private final BzlLoadFunction bzlLoadFunctionForInlining;
  private static final PackageIdentifier rootPackage = PackageIdentifier.createInMainRepo("");

  public WorkspaceFileFunction(
      RuleClassProvider ruleClassProvider,
      PackageFactory packageFactory,
      BlazeDirectories directories,
      BzlLoadFunction bzlLoadFunctionForInlining) {
    this.packageFactory = packageFactory;
    this.directories = directories;
    this.ruleClassProvider = ruleClassProvider;
    this.bzlLoadFunctionForInlining = bzlLoadFunctionForInlining;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws WorkspaceFileFunctionException, InterruptedException {
    WorkspaceFileKey key = (WorkspaceFileKey) skyKey.argument();
    RootedPath workspaceFile = key.getPath();
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    Optional<RootedPath> resolvedFile =
        RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE.get(env);
    if (resolvedFile == null) {
      return null;
    }
    String newWorkspaceFileContents = null;
    FileValue workspaceFileValue = null;
    if (resolvedFile.isPresent()) {
      newWorkspaceFileContents = workspaceFromResolvedValue(resolvedFile.get(), env);
      if (newWorkspaceFileContents == null) {
        return null;
      }
    } else {
      workspaceFileValue = (FileValue) env.getValue(FileValue.key(workspaceFile));
      if (workspaceFileValue == null) {
        return null;
      }
    }
    if (env.valuesMissing()) {
      return null;
    }

    FileOptions options =
        FileOptions.builder()
            // Repository declarations in WORKSPACE have side effects on
            // the set of valid load labels, so load statements cannot all
            // be migrated to the top of the file.
            .requireLoadStatementsFirst(false)
            // Top-level rebinding is permitted because historically
            // WORKSPACE files followed BUILD norms, but this should
            // probably be flipped.
            .allowToplevelRebinding(true)
            .restrictStringEscapes(
                starlarkSemantics.getBool(
                    BuildLanguageOptions.INCOMPATIBLE_RESTRICT_STRING_ESCAPES))
            .build();

    Path repoWorkspace = workspaceFile.getRoot().getRelative(workspaceFile.getRootRelativePath());
    StarlarkFile file;
    try {
      file =
          StarlarkFile.parse(
              ParserInput.fromString(
                  ruleClassProvider.getDefaultWorkspacePrefix(), "/DEFAULT.WORKSPACE"),
              options);
      if (!file.ok()) {
        Event.replayEventsOn(env.getListener(), file.errors());
        throw resolvedValueError("Failed to parse default WORKSPACE file");
      }
      if (newWorkspaceFileContents != null) {
        file =
            StarlarkFile.parseWithPrelude(
                ParserInput.fromString(
                    newWorkspaceFileContents, resolvedFile.get().asPath().toString()),
                file.getStatements(),
                // The WORKSPACE.resolved file breaks through the usual privacy mechanism.
                options.toBuilder().allowLoadPrivateSymbols(true).build());
      } else if (workspaceFileValue.exists()) {
        byte[] bytes =
            FileSystemUtils.readWithKnownFileSize(repoWorkspace, repoWorkspace.getFileSize());
        file =
            StarlarkFile.parseWithPrelude(
                ParserInput.fromLatin1(bytes, repoWorkspace.toString()),
                file.getStatements(),
                options);
        if (!file.ok()) {
          Event.replayEventsOn(env.getListener(), file.errors());
          throw resolvedValueError("Failed to parse WORKSPACE file");
        }
      }

      String suffix;
      if (resolvedFile.isPresent()) {
        suffix = "";
      } else {
        suffix = ruleClassProvider.getDefaultWorkspaceSuffix();
      }

      file =
          StarlarkFile.parseWithPrelude(
              ParserInput.fromString(suffix, "/DEFAULT.WORKSPACE.SUFFIX"),
              file.getStatements(),
              // The DEFAULT.WORKSPACE.SUFFIX file breaks through the usual privacy mechanism.
              options.toBuilder().allowLoadPrivateSymbols(true).build());
      if (!file.ok()) {
        Event.replayEventsOn(env.getListener(), file.errors());
        throw resolvedValueError("Failed to parse default WORKSPACE file suffix");
      }
    } catch (IOException ex) { // TODO(adonovan): reduce try scope
      throw new WorkspaceFileFunctionException(ex, Transience.TRANSIENT);
    }

    // -- end   of historical WorkspaceASTFunction --
    // -- start of historical WorkspaceFileFunction --
    // TODO(adonovan): reorganize and simplify.

    ImmutableList<StarlarkFile> asts = splitAST(file);

    Package.Builder builder =
        packageFactory.newExternalPackageBuilder(
            workspaceFile, ruleClassProvider.getRunfilesPrefix(), starlarkSemantics);

    if (asts.isEmpty()) {
      return new WorkspaceFileValue(
          buildAndReportEvents(builder, env),
          /* loadedModules = */ ImmutableMap.<String, Module>of(),
          /* loadToChunkMap = */ ImmutableMap.<String, Integer>of(),
          /* bindings = */ ImmutableMap.<String, Object>of(),
          workspaceFile,
          /* idx = */ 0, // first fragment
          /* hasNext = */ false,
          ImmutableMap.of(),
          ImmutableSortedSet.of());
    }
    WorkspaceFactory parser;
    WorkspaceFileValue prevValue = null;
    try (Mutability mutability = Mutability.create("workspace", workspaceFile)) {
      parser =
          new WorkspaceFactory(
              builder,
              ruleClassProvider,
              packageFactory.getEnvironmentExtensions(),
              mutability,
              key.getIndex() == 0,
              directories.getEmbeddedBinariesRoot(),
              directories.getWorkspace(),
              directories.getLocalJavabase(),
              starlarkSemantics);
      if (key.getIndex() > 0) {
        prevValue =
            (WorkspaceFileValue)
                env.getValue(WorkspaceFileValue.key(workspaceFile, key.getIndex() - 1));
        if (prevValue == null) {
          return null;
        }
        if (prevValue.next() == null) {
          return prevValue;
        }
        parser.setParent(
            prevValue.getPackage(), prevValue.getLoadedModules(), prevValue.getBindings());
      }
      StarlarkFile ast = asts.get(key.getIndex());
      PackageFunction.BzlLoadResult bzlLoadResult =
          PackageFunction.fetchLoadsFromBuildFile(
              workspaceFile,
              rootPackage,
              /*repoMapping=*/ ImmutableMap.of(),
              ast,
              /*preludeLabel=*/ null,
              key.getIndex(),
              env,
              bzlLoadFunctionForInlining);
      if (bzlLoadResult == null) {
        return null;
      }
      parser.execute(ast, bzlLoadResult.loadedModules, key);
    } catch (NoSuchPackageException e) {
      throw new WorkspaceFileFunctionException(e, Transience.PERSISTENT);
    } catch (NameConflictException e) {
      throw new WorkspaceFileFunctionException(e, Transience.PERSISTENT);
    }

    return new WorkspaceFileValue(
        buildAndReportEvents(builder, env),
        parser.getLoadedModules(),
        createLoadToChunkMap(prevValue, parser, key),
        parser.getVariableBindings(),
        workspaceFile,
        key.getIndex(),
        key.getIndex() < asts.size() - 1,
        ImmutableMap.copyOf(parser.getManagedDirectories()),
        parser.getDoNotSymlinkInExecrootPaths());
  }

  private static Package buildAndReportEvents(Package.Builder pkgBuilder, Environment env)
      throws WorkspaceFileFunctionException {
    Package result;
    try {
      result = pkgBuilder.build();
    } catch (NoSuchPackageException e) {
      throw new WorkspaceFileFunctionException(e, Transience.TRANSIENT);
    }

    Event.replayEventsOn(env.getListener(), pkgBuilder.getEvents());
    for (Postable postable : pkgBuilder.getPosts()) {
      env.getListener().post(postable);
    }

    return result;
  }

  /**
   * This returns a map from load statement to the chunk the load statement originated from.
   *
   * <p>For example, if the WORKSPACE file looked like the following:
   *
   * <pre>
   * load(":a.bzl", "a")
   * x = 0
   * load(":b.bzl", "b")
   * x = 1
   * load(":a.bzl", "a1")
   * load(":c.bzl", "c")
   * x = 2
   * </pre>
   *
   * Then the map for chunk 0 would be {@code {":a.bzl" : 0}}, for chunk 1 it'd be: {@code {":a.bzl"
   * : 0, ":b.bzl" : 1}}, and for chunk 2 it'd be: {@code {":a.bzl" : 0, ":b.bzl" : 1, ":c.bzl" :
   * 2}}
   */
  private static ImmutableMap<String, Integer> createLoadToChunkMap(
      WorkspaceFileValue prevValue, WorkspaceFactory parser, WorkspaceFileKey key) {
    ImmutableMap.Builder<String, Integer> builder = new ImmutableMap.Builder<String, Integer>();
    if (prevValue == null) {
      for (String loadString : parser.getLoadedModules().keySet()) {
        builder.put(loadString, key.getIndex());
      }
    } else {
      builder.putAll(prevValue.getLoadToChunkMap());
      for (String label : parser.getLoadedModules().keySet()) {
        if (!prevValue.getLoadToChunkMap().containsKey(label)) {
          builder.put(label, key.getIndex());
        }
      }
    }
    return builder.build();
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class WorkspaceFileFunctionException extends SkyFunctionException {
    WorkspaceFileFunctionException(NoSuchPackageException e, Transience transience) {
      super(e, transience);
    }

    WorkspaceFileFunctionException(NameConflictException e, Transience transience) {
      super(e, transience);
    }

    WorkspaceFileFunctionException(IOException e, Transience transience) {
      super(e, transience);
    }
  }

  private static WorkspaceFileFunctionException resolvedValueError(String message) {
    return new WorkspaceFileFunctionException(
        new BuildFileContainsErrorsException(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, message),
        Transience.PERSISTENT);
  }

  /**
   * Return the contents of the WORKSPACE file that is implicitly represented by the resolved value
   * found in the given file.
   *
   * <p>TODO(aehlig): at the moment we serialize the value as a string just to re-parse it
   * immediately again; we probably should construct the statements directly out of the value to
   * improve performance.
   */
  private static String workspaceFromResolvedValue(RootedPath resolvedPath, Environment env)
      throws WorkspaceFileFunctionException, InterruptedException {
    ResolvedFileValue resolvedValue =
        (ResolvedFileValue) env.getValue(ResolvedFileValue.key(resolvedPath));
    if (resolvedValue == null) {
      return null;
    }
    List<Map<String, Object>> resolved = resolvedValue.getResolvedValue();
    StringBuilder builder = new StringBuilder();
    for (Map<String, Object> entry : resolved) {
      Object repositories = entry.get(REPOSITORIES);
      if (repositories != null) {
        if (!(repositories instanceof List)) {
          throw resolvedValueError(
              "In 'resolved' the " + REPOSITORIES + " entry is or not a list for item " + entry);
        }
        for (Object repo : (List) repositories) {
          if (!(repo instanceof Map)) {
            throw resolvedValueError("A description of an individual repository is not a map");
          }
          Object rule = ((Map) repo).get(RULE_CLASS);
          if (!(rule instanceof String)) {
            throw resolvedValueError("Expected " + RULE_CLASS + " to be a string.");
          }
          int separatorPosition = ((String) rule).lastIndexOf('%');
          if (separatorPosition < 0) {
            throw resolvedValueError("Malformed rule class: " + ((String) rule));
          }
          String fileName = ((String) rule).substring(0, separatorPosition);
          String symbol = ((String) rule).substring(separatorPosition + 1);

          Object args = ((Map) repo).get(ATTRIBUTES);
          if (!(args instanceof Map)) {
            throw resolvedValueError("Arguments for " + ((String) rule) + " not a dict.");
          }

          builder
              .append("load(\"")
              .append(fileName)
              .append("\", \"")
              .append(symbol)
              .append("\")\n");
          builder.append(symbol).append("(\n");
          for (Map.Entry<?, ?> arg : ((Map<?, ?>) args).entrySet()) {
            Object key = arg.getKey();
            if (!(key instanceof String)) {
              throw resolvedValueError(
                  "In arguments to " + ((String) rule) + " found a non-string key.");
            }
            builder.append("    ").append((String) key).append(" = ");
            builder.append(Starlark.repr(arg.getValue()));
            builder.append(",\n");
          }
          builder.append(")\n\n");
        }
      }
      Object nativeEntry = entry.get(NATIVE);
      if (nativeEntry != null) {
        if (!(nativeEntry instanceof String)) {
          throw resolvedValueError(
              "In 'resolved' the " + NATIVE + " entry is not a string for item " + entry);
        }
        builder.append(nativeEntry).append("\n");
      }
    }
    return builder.toString();
  }

  /**
   * Cut {@code ast} into a list of AST separated by load statements. We cut right before each load
   * statement series.
   */
  // (visible to test)
  static ImmutableList<StarlarkFile> splitAST(StarlarkFile ast) {
    ImmutableList.Builder<StarlarkFile> asts = ImmutableList.builder();
    int prevIdx = 0;
    boolean lastIsLoad = true; // don't cut if the first statement is a load.
    List<Statement> statements = ast.getStatements();
    for (int idx = 0; idx < statements.size(); idx++) {
      Statement st = statements.get(idx);
      if (st instanceof LoadStatement) {
        if (!lastIsLoad) {
          asts.add(ast.subTree(prevIdx, idx));
          prevIdx = idx;
        }
        lastIsLoad = true;
      } else {
        lastIsLoad = false;
      }
    }
    if (!statements.isEmpty()) {
      asts.add(ast.subTree(prevIdx, statements.size()));
    }
    return asts.build();
  }
}
