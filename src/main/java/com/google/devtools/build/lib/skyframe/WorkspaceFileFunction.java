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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFactory;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.packages.WorkspaceFileValue.WorkspaceFileKey;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread.Extension;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A SkyFunction to parse WORKSPACE files.
 */
public class WorkspaceFileFunction implements SkyFunction {

  private final PackageFactory packageFactory;
  private final BlazeDirectories directories;
  private final RuleClassProvider ruleClassProvider;
  private final SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining;
  private static final PackageIdentifier rootPackage = PackageIdentifier.createInMainRepo("");

  public WorkspaceFileFunction(
      RuleClassProvider ruleClassProvider,
      PackageFactory packageFactory,
      BlazeDirectories directories,
      SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining) {
    this.packageFactory = packageFactory;
    this.directories = directories;
    this.ruleClassProvider = ruleClassProvider;
    this.skylarkImportLookupFunctionForInlining = skylarkImportLookupFunctionForInlining;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws WorkspaceFileFunctionException, InterruptedException {

    WorkspaceFileKey key = (WorkspaceFileKey) skyKey.argument();
    RootedPath workspaceRoot = key.getPath();
    WorkspaceASTValue workspaceASTValue = (WorkspaceASTValue) env.getValue(
        WorkspaceASTValue.key(workspaceRoot));
    if (workspaceASTValue == null) {
      return null;
    }
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    RootedPath repoWorkspace =
        RootedPath.toRootedPath(workspaceRoot.getRoot(), workspaceRoot.getRootRelativePath());
    Package.Builder builder =
        packageFactory.newExternalPackageBuilder(
            repoWorkspace, ruleClassProvider.getRunfilesPrefix(), starlarkSemantics);

    if (workspaceASTValue.getASTs().isEmpty()) {
      try {
        return new WorkspaceFileValue(
            /* pkg = */ builder.build(),
            /* importMap = */ ImmutableMap.<String, Extension>of(),
            /* importToChunkMap = */ ImmutableMap.<String, Integer>of(),
            /* bindings = */ ImmutableMap.<String, Object>of(),
            workspaceRoot,
            /* idx = */ 0, // first fragment
            /* hasNext = */ false,
            ImmutableMap.of(),
            ImmutableSortedSet.of());
      } catch (NoSuchPackageException e) {
        throw new WorkspaceFileFunctionException(e, Transience.TRANSIENT);
      }
    }
    WorkspaceFactory parser;
    WorkspaceFileValue prevValue = null;
    try (Mutability mutability = Mutability.create("workspace", repoWorkspace)) {
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
                env.getValue(WorkspaceFileValue.key(key.getPath(), key.getIndex() - 1));
        if (prevValue == null) {
          return null;
        }
        if (prevValue.next() == null) {
          return prevValue;
        }
        parser.setParent(prevValue.getPackage(), prevValue.getImportMap(), prevValue.getBindings());
      }
      StarlarkFile ast = workspaceASTValue.getASTs().get(key.getIndex());
      PackageFunction.SkylarkImportResult importResult =
          PackageFunction.fetchImportsFromBuildFile(
              repoWorkspace,
              rootPackage,
              /*repoMapping=*/ ImmutableMap.of(),
              ast,
              key.getIndex(),
              env,
              skylarkImportLookupFunctionForInlining);
      if (importResult == null) {
        return null;
      }
      parser.execute(ast, importResult.importMap, key);
    } catch (NoSuchPackageException e) {
      throw new WorkspaceFileFunctionException(e, Transience.PERSISTENT);
    } catch (NameConflictException e) {
      throw new WorkspaceFileFunctionException(e, Transience.PERSISTENT);
    }

    try {
      return new WorkspaceFileValue(
          builder.build(),
          parser.getImportMap(),
          createImportToChunkMap(prevValue, parser, key),
          parser.getVariableBindings(),
          workspaceRoot,
          key.getIndex(),
          key.getIndex() < workspaceASTValue.getASTs().size() - 1,
          ImmutableMap.copyOf(parser.getManagedDirectories()),
          parser.getDoNotSymlinkInExecrootPaths());
    } catch (NoSuchPackageException e) {
      throw new WorkspaceFileFunctionException(e, Transience.TRANSIENT);
    }
  }

  /**
   * This returns a map from import statement to the chunk the
   * import statement originated from.
   *
   * For example, if the WORKSPACE file looked like the following:
   * load(":a.bzl", "a")
   * x = 0
   * load(":b.bzl", "b")
   * x = 1
   * load(":a.bzl", "a1")
   * load(":c.bzl", "c")
   * x = 2
   *
   * Then the map for chunk 0 would be: {@code {":a.bzl" : 0}}
   * for chunk 1 would be: {@code {":a.bzl" : 0, ":b.bzl" : 1}}
   * for chunk 2 would be: {@code {":a.bzl" : 0, ":b.bzl" : 1, ":c.bzl" : 2}}
   */
  private ImmutableMap<String, Integer> createImportToChunkMap(
      WorkspaceFileValue prevValue, WorkspaceFactory parser, WorkspaceFileKey key) {
    ImmutableMap.Builder<String, Integer> builder = new ImmutableMap.Builder<String, Integer>();
    if (prevValue == null) {
      Map<String, Integer> map =
          parser.getImportMap().keySet().stream()
              .collect(Collectors.toMap(Function.identity(), s -> key.getIndex()));
      builder.putAll(map);
    } else {
      builder.putAll(prevValue.getImportToChunkMap());
      for (String label : parser.getImportMap().keySet()) {
        if (!prevValue.getImportToChunkMap().containsKey(label)) {
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
}
