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
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * Scans source files to determine the bounding set of transitively referenced include files.
 *
 * <p>Include scanning is performance-critical code since it has to parse a lot of C++ code, mostly
 * on the same machine where Blaze runs.
 *
 * <p>Include scanning works by first adding all the potential header files to the "scheduling
 * dependencies" of the action. This makes Skyframe build these files (if they are generated) before
 * the action is executed. Note that this means that the "inputs" of the action as seen by Skyframe
 * are different from what {@code Action.getInputs()} returns: the former includes scheduling
 * dependencies, whereas the latter does not.
 *
 * <p>Then {@code Action.discoverInputs()} is called, which then runs the include scanning machinery
 * and eventually calls {@code Action.updateInputs()}. That method in turn adds the discovered
 * inputs to what {@code getInputs()} returns. It's implemented in a separate method because when
 * the action is a local action cache hit, the discovered inputs of the action are read from the
 * local action cache and added to the action's inputs by calling {@code updateInputs()} without
 * calling {@code discoverInputs()}.
 *
 * <p>The include scanner consists of two parts:
 *
 * <ol>
 *   <li>The part that parses the source files and extracts the include directives.
 *   <li>The logic that evaluates the include directives and recursively parses the referenced
 *       files.
 * </ol>
 *
 * <p>Parsing source files can be done in two ways: locally (in {@code IncludeParser}) and remotely
 * (using {@code GrepIncludesAction}). The latter is useful when parsing generated source files: if
 * the file is large, it's beneficial not to have to shuttle the file from the remote execution
 * cluster to Bazel. In either case, the result of parsing is a list of include directives, which
 * are essentially a pair of (include style, include path), where the style is the Cartesian product
 * of "quote"/"angle" and "regular include"/"include next".
 *
 * <p>This parsing is very simplistic: it doesn't run an actual preprocessor, only parses its
 * directives. This means that computed includes cannot possibly be handled, but otherwise, it's an
 * overestimate of the actual headers used (for example, it takes both branches of an {@code #if})
 * directive. This works because if an unused file is handed over to the compile action, it's
 * suboptimal but still results in a successful compilation.
 *
 * <p>Computed includes are handled by adding hints to the include scanner. This is implemented in
 * {@code IncludeHintsFunction} which is short-circuited in Bazel (not at Google, though)
 *
 * <p>Evaluating the include directives is implemented in {@code LegacyIncludeScanner}, which,
 * despite its name is not really legacy. Notably, it maintains a cache of the results of its work.
 * Its key is not simply the file processed, but the file processed and the set of include
 * directories used (because the latter can obviously affect the result). This cache is flushed on
 * every Bazel command.
 *
 * <p>After the include scanner is done, the resulting inputs are handed over to the regular action
 * execution machinery. Once the action is executed, the .d file produced by the compiler (or the
 * output of the {@code /showIncludes} command line flag when using MSVC) is parsed to figure out
 * which headers were actually used. This is implemented in {@code discoverInputsFromDotdFiles()}
 * and {@code discoverInputsFromShowIncludes()} in {@code CppCompileAction}. Then the result of this
 * is used to remove the headers from the inputs of the action that the compiler didn't end up
 * using.
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
   *
   * <ol>
   *   <li>We compile a header module - there, the .cppmap file is the main source file (which we do
   *       not include-scan, as that would require an extra parser), and thus already in the input;
   *       all headers in the .cppmap file are our entry points for include scanning, but are not
   *       yet in the inputs - they get added here.
   *   <li>We compile an object file that uses a header module; currently using a header module
   *       requires all headers it can reference to be available for the compilation. The header
   *       module can reference headers that are not in the transitive include closure of the
   *       current translation unit. Therefore, {@link CppCompileAction} adds all headers specified
   *       transitively for compiled header modules as include scanning entry points, and we need to
   *       add the entry points to the inputs here.
   * </ol>
   *
   * <p>{@code mainSource} is the source file relative to which the {@code cmdlineIncludes} are
   * interpreted.
   *
   * <p>Additional dependencies may be requested via {@link
   * ActionExecutionContext#getEnvironmentForDiscoveringInputs}. If any dependency is not
   * immediately available, processing will be short-circuited. The caller should check {@link
   * com.google.devtools.build.skyframe.SkyFunction.Environment#valuesMissing} - if it returns
   * {@code true}, then include scanning did not complete and a skyframe restart is necessary.
   *
   * @throws NoSuchPackageException if hint collection fails due to package problems
   */
  void processAsync(
      Artifact mainSource,
      Collection<Artifact> sources,
      IncludeScanningHeaderData includeScanningHeaderData,
      List<String> cmdlineIncludes,
      Set<Artifact> includes,
      ActionExecutionMetadata actionExecutionMetadata,
      ActionExecutionContext actionExecutionContext,
      Artifact grepIncludes,
      @Nullable PlatformInfo grepIncludesExecutionPlatform)
      throws IOException, NoSuchPackageException, ExecException, InterruptedException;

  /**
   * Holds pre-aggregated information that the {@link IncludeScanner} needs from the compilation
   * action.
   */
  final class IncludeScanningHeaderData {
    /**
     * Lookup table to find the {@link Artifact}s of generated files based on their {@link
     * Artifact#getExecPath}.
     */
    private final Map<PathFragment, Artifact> pathToDeclaredHeader;

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

    /**
     * Tests whether the given artifact is a valid header even if it is not declared, i.e. a
     * transitive dependency. If null, assume all headers can be included.
     */
    @Nullable private final Predicate<Artifact> isValidUndeclaredHeader;

    public IncludeScanningHeaderData(
        Map<PathFragment, Artifact> pathToDeclaredHeader,
        Set<Artifact> modularHeaders,
        List<PathFragment> systemIncludeDirs,
        List<String> cmdlineIncludes,
        @Nullable Predicate<Artifact> isValidUndeclaredHeader) {
      this.pathToDeclaredHeader = pathToDeclaredHeader;
      this.modularHeaders = modularHeaders;
      this.systemIncludeDirs = systemIncludeDirs;
      this.cmdlineIncludes = cmdlineIncludes;
      this.isValidUndeclaredHeader = isValidUndeclaredHeader;
    }

    public boolean isDeclaredHeader(PathFragment header) {
      return pathToDeclaredHeader.containsKey(header);
    }

    public Artifact getHeaderArtifact(PathFragment header) {
      return pathToDeclaredHeader.get(header);
    }

    public boolean isModularHeader(Artifact header) {
      return modularHeaders.contains(header);
    }

    public List<PathFragment> getSystemIncludeDirs() {
      return systemIncludeDirs;
    }

    public List<String> getCmdlineIncludes() {
      return cmdlineIncludes;
    }

    public boolean isLegalHeader(Artifact header) {
      return isValidUndeclaredHeader == null
          || pathToDeclaredHeader.containsKey(header.getExecPath())
          || isValidUndeclaredHeader.test(header);
    }

    public static class Builder {
      private final Map<PathFragment, Artifact> pathToDeclaredHeader;
      private final Set<Artifact> modularHeaders;
      private List<PathFragment> systemIncludeDirs = ImmutableList.of();
      private List<String> cmdlineIncludes = ImmutableList.of();
      @Nullable private Predicate<Artifact> isValidUndeclaredHeader = null;

      public Builder(
          Map<PathFragment, Artifact> pathToDeclaredHeader, Set<Artifact> modularHeaders) {
        this.pathToDeclaredHeader = pathToDeclaredHeader;
        this.modularHeaders = modularHeaders;
      }

      @CanIgnoreReturnValue
      public Builder setSystemIncludeDirs(List<PathFragment> systemIncludeDirs) {
        this.systemIncludeDirs = systemIncludeDirs;
        return this;
      }

      @CanIgnoreReturnValue
      public Builder setCmdlineIncludes(List<String> cmdlineIncludes) {
        this.cmdlineIncludes = cmdlineIncludes;
        return this;
      }

      @CanIgnoreReturnValue
      public Builder setIsValidUndeclaredHeader(
          @Nullable Predicate<Artifact> isValidUndeclaredHeader) {
        this.isValidUndeclaredHeader = isValidUndeclaredHeader;
        return this;
      }

      public IncludeScanningHeaderData build() {
        return new IncludeScanningHeaderData(
            pathToDeclaredHeader,
            modularHeaders,
            systemIncludeDirs,
            cmdlineIncludes,
            isValidUndeclaredHeader);
      }
    }
  }
}
