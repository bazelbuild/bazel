// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.includescanning;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CppIncludeScanningContext;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.IncludeScanning.Code;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Include scanning context implementation. */
public final class CppIncludeScanningContextImpl implements CppIncludeScanningContext {
  private final Supplier<IncludeScannerSupplier> includeScannerSupplier;

  public CppIncludeScanningContextImpl(Supplier<IncludeScannerSupplier> includeScannerSupplier) {
    this.includeScannerSupplier = includeScannerSupplier;
  }

  @Override
  @Nullable
  public List<Artifact> findAdditionalInputs(
      CppCompileAction action,
      ActionExecutionContext actionExecutionContext,
      IncludeScanningHeaderData includeScanningHeaderData)
      throws ExecException, InterruptedException {
    Preconditions.checkNotNull(includeScannerSupplier, action);

    Set<Artifact> includes = Sets.newConcurrentHashSet();
    includes.addAll(action.getBuiltInIncludeFiles());

    // Deduplicate include directories. This can occur especially with "built-in" and "system"
    // include directories because of the way we retrieve them. Duplicate include directories
    // really mess up #include_next directives.
    Set<PathFragment> includeDirs = new LinkedHashSet<>(action.getIncludeDirs());
    List<PathFragment> quoteIncludeDirs = action.getQuoteIncludeDirs();
    List<PathFragment> frameworkIncludeDirs = action.getFrameworkIncludeDirs();
    List<String> cmdlineIncludes = includeScanningHeaderData.getCmdlineIncludes();

    includeDirs.addAll(includeScanningHeaderData.getSystemIncludeDirs());

    // Add the system include paths to the list of include paths.
    List<PathFragment> absoluteBuiltInIncludeDirs = new ArrayList<>();
    for (PathFragment pathFragment : action.getBuiltInIncludeDirectories()) {
      if (pathFragment.isAbsolute()) {
        absoluteBuiltInIncludeDirs.add(pathFragment);
      }
      includeDirs.add(pathFragment);
    }

    ImmutableList<PathFragment> includeDirList = ImmutableList.copyOf(includeDirs);
    IncludeScanner scanner =
        includeScannerSupplier
            .get()
            .scannerFor(quoteIncludeDirs, includeDirList, frameworkIncludeDirs);

    Artifact mainSource = action.getMainIncludeScannerSource();
    ImmutableList<Artifact> sources =
        expandTreeArtifacts(
            action.getIncludeScannerSources(),
            actionExecutionContext.getEnvironmentForDiscoveringInputs());
    if (sources == null) {
      return null;
    }

    try (SilentCloseable c =
        Profiler.instance()
            .profile(ProfilerTask.SCANNER, action.getSourceFile().getExecPathString())) {
      scanner.processAsync(
          mainSource,
          sources,
          includeScanningHeaderData,
          cmdlineIncludes,
          includes,
          action,
          actionExecutionContext,
          action.getGrepIncludes(),
          action.getExecutionPlatform());
      if (actionExecutionContext.getEnvironmentForDiscoveringInputs().valuesMissing()) {
        return null;
      }
      return collect(actionExecutionContext, includes, absoluteBuiltInIncludeDirs);
    } catch (IOException e) {
      throw new EnvironmentalExecException(
          e, createFailureDetail("Include scanning IOException", Code.SCANNING_IO_EXCEPTION));
    } catch (NoSuchPackageException e) {
      throw new EnvironmentalExecException(
          e,
          createFailureDetail(
              "Error for BUILD file during include scanning: " + e.getMessage(),
              Code.PACKAGE_LOAD_FAILURE));
    }
  }

  /**
   * Returns a list of artifacts with all tree artifacts replaced by their expansion or null if we
   * are missing a Skyframe dependency.
   *
   * <p>We take an ad-hoc approach, which consults Skyframe to retrieve the tree expansions. This is
   * necessary because include scanning may include tree artifacts which are not inputs of the
   * original action.
   *
   * <p>Normally, we expand tree artifacts using a tree artifact expander from the {@link
   * ActionExecutionContext}, however the expander available before include scanning only captures
   * tree artifacts from the action inputs, which is insufficient. In fact, the action execution
   * context for include scanning has a null expander in it.
   */
  @Nullable
  private static ImmutableList<Artifact> expandTreeArtifacts(
      ImmutableList<Artifact> artifacts, Environment env) throws InterruptedException {
    ImmutableList<Artifact> trees =
        artifacts.stream().filter(Artifact::isTreeArtifact).collect(toImmutableList());
    if (trees.isEmpty()) {
      return artifacts;
    }

    SkyframeLookupResult expansions = env.getValuesAndExceptions(trees);
    ImmutableList.Builder<Artifact> expanded = ImmutableList.builder();
    for (var artifact : artifacts) {
      if (!artifact.isTreeArtifact()) {
        expanded.add(artifact);
        continue;
      }

      TreeArtifactValue treeArtifactValue = (TreeArtifactValue) expansions.get(artifact);
      if (treeArtifactValue == null) {
        return null;
      }
      expanded.addAll(treeArtifactValue.getChildren());
    }
    return expanded.build();
  }

  private static List<Artifact> collect(
      ActionExecutionContext actionExecutionContext,
      Set<Artifact> includes,
      List<PathFragment> absoluteBuiltInIncludeDirs)
      throws ExecException {
    // Collect inputs and output
    List<Artifact> inputs = new ArrayList<>(includes.size());
    for (Artifact included : includes) {
      // Check for absolute includes -- we assign the file system root as
      // the root path for such includes
      if (included.getRoot().getRoot().isAbsolute()) {
        if (FileSystemUtils.startsWithAny(
            actionExecutionContext.getInputPath(included).asFragment(),
            absoluteBuiltInIncludeDirs)) {
          // Skip include files found in absolute include directories.
          continue;
        }
        throw new UserExecException(
            createFailureDetail(
                "illegal absolute path to include file: "
                    + actionExecutionContext.getInputPath(included),
                Code.ILLEGAL_ABSOLUTE_PATH));
      }
      if (included.hasParent() && included.getParent().isTreeArtifact()) {
        // Note that this means every file in the TreeArtifact becomes an input to the action, and
        // we have spurious rebuilds if non-included files change.
        Preconditions.checkArgument(
            included instanceof TreeFileArtifact, "Not a TreeFileArtifact: %s", included);
        inputs.add(included.getParent());
      } else {
        inputs.add(included);
      }
    }
    return inputs;
  }

  private static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setIncludeScanning(FailureDetails.IncludeScanning.newBuilder().setCode(detailedCode))
        .build();
  }
}
