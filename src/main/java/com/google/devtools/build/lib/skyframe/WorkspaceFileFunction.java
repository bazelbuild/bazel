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
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.WorkspaceFactory;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.packages.WorkspaceFileValue.WorkspaceFileKey;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;

/** A SkyFunction to parse the WORKSPACE file at a given path. */
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
    WorkspaceASTValue workspaceASTValue =
        (WorkspaceASTValue) env.getValue(WorkspaceASTValue.key(workspaceFile));
    if (workspaceASTValue == null) {
      return null;
    }
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    Package.Builder builder =
        packageFactory.newExternalPackageBuilder(
            workspaceFile, ruleClassProvider.getRunfilesPrefix(), starlarkSemantics);

    if (workspaceASTValue.getASTs().isEmpty()) {
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
      StarlarkFile ast = workspaceASTValue.getASTs().get(key.getIndex());
      ImmutableMap<String, Module> loadedModules =
          PackageFunction.fetchLoadsFromBuildFile(
              workspaceFile,
              rootPackage,
              /*repoMapping=*/ ImmutableMap.of(),
              ast,
              key.getIndex(),
              env,
              bzlLoadFunctionForInlining);
      if (loadedModules == null) {
        return null;
      }
      parser.execute(ast, loadedModules, key);
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
        key.getIndex() < workspaceASTValue.getASTs().size() - 1,
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
}
