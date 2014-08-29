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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionMetadata;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Scans source files to determine the bounding set of transitively referenced include files.
 *
 * <p>Note that include scanning is performance-critical code.
 */
public interface IncludeScanner {
  /**
   * Processes a source file and a list of includes extracted from command line
   * flags. Adds all found files to the provided set {@code includes}. This
   * method takes into account the path- and file-level hints that are part of
   * this include scanner.
   */
  public void process(Path source, Map<Path, Path> legalOutputPaths,
      List<String> cmdlineIncludes, Set<Path> includes,
      ActionExecutionContext actionExecutionContext, ActionMetadata owner)
      throws IOException, ExecException, InterruptedException;

  /** Supplies IncludeScanners upon request. */
  interface IncludeScannerSupplier {
    /** Returns the possibly shared scanner to be used for a given pair of include paths. */
    IncludeScanner scannerFor(List<Path> quoteIncludePaths, List<Path> includePaths,
        RemoteIncludeExtractor remoteScanner);
  }

  /**
   * Helper class that exists just to provide a static method that prepares the arguments with which
   * to call an IncludeScanner.
   */
  class IncludeScanningPreparer {
    private IncludeScanningPreparer() {}

    /**
     * Returns the files transitively included by the source files of the given IncludeScannable.
     *
     * @param action IncludeScannable whose sources' transitive includes will be returned.
     * @param includeScannerSupplier supplies IncludeScanners to actually do the transitive scanning
     *                               (and caching results) for a given source file.
     * @param actionExecutionContext the context for {@code action}.
     * @param profilerTaskName what the {@link Profiler} should record this call for.
     * @param ownerActionMetadata the owner to be associated with this scan.
     */
    public static List<String> scanForIncludedInputs(IncludeScannable action,
        IncludeScannerSupplier includeScannerSupplier,
        ActionExecutionContext actionExecutionContext,
        String profilerTaskName, ActionMetadata ownerActionMetadata)
            throws ExecException, InterruptedException {

      Set<Path> includes = Sets.newConcurrentHashSet();

      Executor executor = actionExecutionContext.getExecutor();
      Path execRoot = executor.getExecRoot();

      RemoteIncludeExtractor remoteScanner = Preconditions.checkNotNull(
          executor.getContext(RemoteIncludeExtractor.class),
          action);
      List<Path> absoluteBuiltInIncludeDirs = new ArrayList<>();

      Profiler profiler = Profiler.instance();
      try {
        profiler.startTask(ProfilerTask.SCANNER, profilerTaskName);

        // We need to scan the action itself, but also the auxiliary scannables
        // (for LIPO). There is no need to call getAuxiliaryScannables
        // recursively.
        for (IncludeScannable scannable :
          Iterables.concat(ImmutableList.of(action), action.getAuxiliaryScannables())) {

          Map<Path, Path> legalOutputPaths = scannable.getLegalGeneratedScannerFileMap();
          List<PathFragment> includeDirs = new ArrayList<>(scannable.getIncludeDirs());
          List<PathFragment> quoteIncludeDirs = scannable.getQuoteIncludeDirs();
          List<String> cmdlineIncludes = scannable.getCmdlineIncludes();

          for (PathFragment pathFragment : scannable.getSystemIncludeDirs()) {
            includeDirs.add(pathFragment);
          }

          // Add the system include paths to the list of include paths.
          for (PathFragment pathFragment : action.getBuiltInIncludeDirectories()) {
            if (pathFragment.isAbsolute()) {
              absoluteBuiltInIncludeDirs.add(execRoot.getRelative(pathFragment));
            }
            includeDirs.add(pathFragment);
          }

          IncludeScanner scanner = includeScannerSupplier.scannerFor(
              relativeTo(execRoot, quoteIncludeDirs),
              relativeTo(execRoot, includeDirs), remoteScanner);

          for (PathFragment sourcePathFragment : scannable.getIncludeScannerSources()) {
            // Make the source file relative to execution root, so that even inclusions
            // found relative to the current file are in the output tree.
            // TODO(bazel-team):  Remove this once relative paths are used during analysis.
            Path sourcePath = execRoot.getRelative(sourcePathFragment);
            scanner.process(sourcePath, legalOutputPaths, cmdlineIncludes, includes,
                actionExecutionContext,
                ownerActionMetadata);
          }
        }
      } catch (IOException e) {
        throw new EnvironmentalExecException(e.getMessage());
      } finally {
        profiler.completeTask(ProfilerTask.SCANNER);
      }

      // Collect inputs and output
      List<String> inputs = new ArrayList<>();
      for (Path included : includes) {
        if (FileSystemUtils.startsWithAny(included, absoluteBuiltInIncludeDirs)) {
          // Skip include files found in absolute include directories. This currently only applies
          // to grte.
          continue;
        }
        if (!included.startsWith(execRoot)) {
          throw new UserExecException("illegal absolute path to include file: " + included);
        }
        inputs.add(included.relativeTo(execRoot).getPathString());
      }
      return inputs;
    }

    private static List<Path> relativeTo(
        Path path, Collection<PathFragment> fragments) {
      List<Path> result = Lists.newArrayListWithCapacity(fragments.size());
      for (PathFragment fragment : fragments) {
        result.add(path.getRelative(fragment));
      }
      return result;
    }
  }
}
