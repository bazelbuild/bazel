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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
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
   * Processes source files and a list of includes extracted from command line flags. Adds all found
   * files to the provided set {@code includes}.
   *
   * <p>The resulting set will include {@code mainSource} and {@code sources}. This has no real
   * impact in the case that we are scanning a single source file, since it is already known to be
   * an input. However, this is necessary when we have more than one source to scan from, for
   * example when building C++ modules. In that case we have one of two possibilities:
   * <ol>
   * <li>We compile a header module - there, the .cppmap file is the main source file (which we do
   *     not include-scan, as that would require an extra parser), and thus already in the input;
   *     all headers in the .cppmap file are our entry points for include scanning, but are not yet
   *     in the inputs - they get added here.</li>
   * <li>We compile an object file that uses a header module; currently using a header module
   *     requires all headers it can reference to be available for the compilation. The header
   *     module can reference headers that are not in the transitive include closure of the current
   *     translation unit. Therefore, {@link CppCompileAction} adds all headers specified
   *     transitively for compiled header modules as include scanning entry points, and we need to
   *     add the entry points to the inputs here.</li></ol>
   * </p>
   *
   * <p>{@code mainSource} is the source file relative to which the {@code cmdlineIncludes} are
   * interpreted.</p>
   */
  void process(
      Artifact mainSource,
      Collection<Artifact> sources,
      IncludeScanningHeaderData includeScanningHeaderData,
      List<String> cmdlineIncludes,
      Set<Artifact> includes,
      ActionExecutionMetadata actionExecutionMetadata,
      ActionExecutionContext actionExecutionContext,
      Artifact grepIncludes)
      throws IOException, ExecException, InterruptedException;

  /** Supplies IncludeScanners upon request. */
  interface IncludeScannerSupplier {
    /**
     * Returns the possibly shared scanner to be used for a given pair of include paths. The paths
     * are specified as PathFragments relative to the execution root.
     */
    IncludeScanner scannerFor(List<PathFragment> quoteIncludePaths,
        List<PathFragment> includePaths);
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
     * @param includeScannerSupplier supplies IncludeScanners to actually do the transitive
     *                               scanning (and caching results) for a given source file.
     * @param actionExecutionContext the context for {@code action}.
     * @param profilerTaskName what the {@link Profiler} should record this call for.
     */
    public static Collection<Artifact> scanForIncludedInputs(
        IncludeScannable action,
        IncludeScannerSupplier includeScannerSupplier,
        IncludeScanningHeaderData includeScanningHeaderData,
        ActionExecutionMetadata actionExecutionMetadata,
        ActionExecutionContext actionExecutionContext,
        String profilerTaskName)
        throws ExecException, InterruptedException {

      Set<Artifact> includes = Sets.newConcurrentHashSet();

      final List<PathFragment> absoluteBuiltInIncludeDirs = new ArrayList<>();
      includes.addAll(action.getBuiltInIncludeFiles());

      Profiler profiler = Profiler.instance();
      try (SilentCloseable c = profiler.profile(ProfilerTask.SCANNER, profilerTaskName)) {
        // Deduplicate include directories. This can occur especially with "built-in" and "system"
        // include directories because of the way we retrieve them. Duplicate include directories
        // really mess up #include_next directives.
        Set<PathFragment> includeDirs = new LinkedHashSet<>(action.getIncludeDirs());
        List<PathFragment> quoteIncludeDirs = action.getQuoteIncludeDirs();
        List<String> cmdlineIncludes = includeScanningHeaderData.getCmdlineIncludes();

        includeDirs.addAll(includeScanningHeaderData.getSystemIncludeDirs());

        // Add the system include paths to the list of include paths.
        for (PathFragment pathFragment : action.getBuiltInIncludeDirectories()) {
          if (pathFragment.isAbsolute()) {
            absoluteBuiltInIncludeDirs.add(pathFragment);
          }
          includeDirs.add(pathFragment);
        }

        List<PathFragment> includeDirList = ImmutableList.copyOf(includeDirs);
        IncludeScanner scanner =
            includeScannerSupplier.scannerFor(quoteIncludeDirs, includeDirList);

        Artifact mainSource = action.getMainIncludeScannerSource();
        Collection<Artifact> sources = action.getIncludeScannerSources();
        scanner.process(
            mainSource,
            sources,
            includeScanningHeaderData,
            cmdlineIncludes,
            includes,
            actionExecutionMetadata,
            actionExecutionContext,
            action.getGrepIncludes());

      } catch (IOException e) {
        throw new EnvironmentalExecException(e.getMessage());
      }

      // Collect inputs and output
      List<Artifact> inputs = new ArrayList<>();
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
              "illegal absolute path to include file: "
                  + actionExecutionContext.getInputPath(included));
        }
        inputs.add(included);
      }
      return inputs;
    }
  }

  /**
   * Holds pre-aggregated information that the {@link IncludeScanner} needs from the compilation
   * action.
   */
  class IncludeScanningHeaderData {
    /**
     * Lookup table to find the {@link Artifact}s of generated files based on their {@link
     * Artifact#execPath}.
     */
    private final Map<PathFragment, Artifact> pathToLegalOutputArtifact;

    /**
     * The set of headers that are modular, i.e. are going to be read as a serialized AST rather
     * than from the textual source file. Depending on the implementation, it is likely that further
     * input discovery through such headers is unnecessary as the serialized AST is self-contained.
     */
    private final Set<Artifact> modularHeaders;

    /**
     * The list of "-isystem" include paths that should be used by the IncludeScanner for this
     * action. The compiler searches these paths ahead of the built-in system include paths, but
     * after all other paths. "-isystem" paths are treated the same as normal system directories.
     */
    private final List<PathFragment> systemIncludeDirs;

    /**
     * A list of "-include" inclusions specified explicitly on the command line of this action. The
     * compiler will imagine that these files have been quote-included at the beginning of each
     * source file.
     */
    private final List<String> cmdlineIncludes;

    public IncludeScanningHeaderData(
        Map<PathFragment, Artifact> pathToLegalOutputArtifact,
        Set<Artifact> modularHeaders,
        List<PathFragment> systemIncludeDirs,
        List<String> cmdlineIncludes) {
      this.pathToLegalOutputArtifact = pathToLegalOutputArtifact;
      this.modularHeaders = modularHeaders;
      this.systemIncludeDirs = systemIncludeDirs;
      this.cmdlineIncludes = cmdlineIncludes;
    }

    public Set<Artifact> getModularHeaders() {
      return modularHeaders;
    }

    public Map<PathFragment, Artifact> getPathToLegalOutputArtifact() {
      return pathToLegalOutputArtifact;
    }

    public List<PathFragment> getSystemIncludeDirs() {
      return systemIncludeDirs;
    }

    public List<String> getCmdlineIncludes() {
      return cmdlineIncludes;
    }

    public static class Builder {
      private final Map<PathFragment, Artifact> pathToLegalOutputArtifact;
      private final Set<Artifact> modularHeaders;
      private List<PathFragment> systemIncludeDirs = ImmutableList.of();
      private List<String> cmdlineIncludes = ImmutableList.of();

      public Builder(
          Map<PathFragment, Artifact> pathToLegalOutputArtifact, Set<Artifact> modularHeaders) {
        this.pathToLegalOutputArtifact = pathToLegalOutputArtifact;
        this.modularHeaders = modularHeaders;
      }

      public Builder setSystemIncludeDirs(List<PathFragment> systemIncludeDirs) {
        this.systemIncludeDirs = systemIncludeDirs;
        return this;
      }

      public Builder setCmdlineIncludes(List<String> cmdlineIncludes) {
        this.cmdlineIncludes = cmdlineIncludes;
        return this;
      }

      public IncludeScanningHeaderData build() {
        return new IncludeScanningHeaderData(
            pathToLegalOutputArtifact, modularHeaders, systemIncludeDirs, cmdlineIncludes);
      }
    }
  }
}
