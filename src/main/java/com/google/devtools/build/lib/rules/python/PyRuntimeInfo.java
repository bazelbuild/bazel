// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.python;

import static net.starlark.java.eval.Starlark.NONE;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.starlarkbuildapi.python.PyRuntimeInfoApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Instance of the provider type that describes Python runtimes.
 *
 * <p>Invariant: Exactly one of {@link #interpreterPath} and {@link #interpreter} is non-null. The
 * former corresponds to a platform runtime, and the latter to an in-build runtime; these two cases
 * are mutually exclusive. In addition, {@link #files} is non-null if and only if {@link
 * #interpreter} is non-null; in other words files are only used for in-build runtimes. These
 * invariants mirror the user-visible API on {@link PyRuntimeInfoApi} except that {@code None} is
 * replaced by null.
 */
public final class PyRuntimeInfo implements Info, PyRuntimeInfoApi<Artifact> {

  /** The Starlark-accessible top-level builtin name for this provider type. */
  public static final String STARLARK_NAME = "PyRuntimeInfo";

  /** The singular {@code PyRuntimeInfo} provider type object. */
  public static final PyRuntimeInfoProvider PROVIDER = new PyRuntimeInfoProvider();

  private final Location location;
  @Nullable private final PathFragment interpreterPath;
  @Nullable private final Artifact interpreter;
  // Validated on initialization to contain Artifact
  @Nullable private final Depset files;
  @Nullable private final Artifact coverageTool;
  @Nullable private final Depset coverageFiles;
  /** Invariant: either PY2 or PY3. */
  private final PythonVersion pythonVersion;

  private final String stubShebang;
  @Nullable private final Artifact bootstrapTemplate;

  private PyRuntimeInfo(
      @Nullable Location location,
      @Nullable PathFragment interpreterPath,
      @Nullable Artifact interpreter,
      @Nullable Depset files,
      @Nullable Artifact coverageTool,
      @Nullable Depset coverageFiles,
      PythonVersion pythonVersion,
      @Nullable String stubShebang,
      @Nullable Artifact bootstrapTemplate) {
    Preconditions.checkArgument((interpreterPath == null) != (interpreter == null));
    Preconditions.checkArgument((interpreter == null) == (files == null));
    Preconditions.checkArgument((coverageTool == null) == (coverageFiles == null));
    Preconditions.checkArgument(pythonVersion.isTargetValue());
    this.location = location != null ? location : Location.BUILTIN;
    this.files = files;
    this.interpreterPath = interpreterPath;
    this.interpreter = interpreter;
    this.coverageTool = coverageTool;
    this.coverageFiles = coverageFiles;
    this.pythonVersion = pythonVersion;
    if (stubShebang != null && !stubShebang.isEmpty()) {
      this.stubShebang = stubShebang;
    } else {
      this.stubShebang = PyRuntimeInfoApi.DEFAULT_STUB_SHEBANG;
    }
    this.bootstrapTemplate = bootstrapTemplate;
  }

  @Override
  public PyRuntimeInfoProvider getProvider() {
    return PROVIDER;
  }

  @Override
  public Location getCreationLocation() {
    return location;
  }

  /** Constructs an instance from native rule logic (built-in location) for an in-build runtime. */
  public static PyRuntimeInfo createForInBuildRuntime(
      Artifact interpreter,
      NestedSet<Artifact> files,
      @Nullable Artifact coverageTool,
      @Nullable NestedSet<Artifact> coverageFiles,
      PythonVersion pythonVersion,
      @Nullable String stubShebang,
      @Nullable Artifact bootstrapTemplate) {
    return new PyRuntimeInfo(
        /* location= */ null,
        /* interpreterPath= */ null,
        interpreter,
        Depset.of(Artifact.class, files),
        coverageTool,
        coverageFiles == null ? null : Depset.of(Artifact.class, coverageFiles),
        pythonVersion,
        stubShebang,
        bootstrapTemplate);
  }

  /** Constructs an instance from native rule logic (built-in location) for a platform runtime. */
  public static PyRuntimeInfo createForPlatformRuntime(
      PathFragment interpreterPath,
      @Nullable Artifact coverageTool,
      @Nullable NestedSet<Artifact> coverageFiles,
      PythonVersion pythonVersion,
      @Nullable String stubShebang,
      @Nullable Artifact bootstrapTemplate) {
    return new PyRuntimeInfo(
        /* location= */ null,
        interpreterPath,
        /* interpreter= */ null,
        /* files= */ null,
        coverageTool,
        coverageFiles == null ? null : Depset.of(Artifact.class, coverageFiles),
        pythonVersion,
        stubShebang,
        bootstrapTemplate);
  }

  @Override
  public boolean equals(Object other) {
    // PyRuntimeInfo implements value equality, but note that it contains identity-equality fields
    // (depsets), so you generally shouldn't rely on equality comparisons.
    if (!(other instanceof PyRuntimeInfo)) {
      return false;
    }
    PyRuntimeInfo otherInfo = (PyRuntimeInfo) other;
    return (this.interpreterPath.equals(otherInfo.interpreterPath)
        && this.interpreter.equals(otherInfo.interpreter)
        && this.files.equals(otherInfo.files)
        && this.coverageTool.equals(otherInfo.coverageTool)
        && this.coverageFiles.equals(otherInfo.coverageFiles)
        && this.stubShebang.equals(otherInfo.stubShebang));
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        PyRuntimeInfo.class,
        interpreterPath,
        interpreter,
        coverageTool,
        coverageFiles,
        files,
        stubShebang);
  }

  /**
   * Returns true if this is an in-build runtime as opposed to a platform runtime -- that is, if
   * this refers to a target within the build as opposed to a path to a system interpreter.
   *
   * <p>{@link #getInterpreter} and {@link #getFiles} are non-null if and only if this is an
   * in-build runtime, whereas {@link #getInterpreterPath} is non-null if and only if this is a
   * platform runtime.
   *
   * <p>Note: It is still possible for an in-build runtime to reference the system interpreter, as
   * in the case where it is a wrapper script.
   */
  public boolean isInBuild() {
    return getInterpreter() != null;
  }

  @Nullable
  public PathFragment getInterpreterPath() {
    return interpreterPath;
  }

  @Override
  @Nullable
  public String getInterpreterPathString() {
    return interpreterPath == null ? null : interpreterPath.getPathString();
  }

  @Override
  @Nullable
  public Artifact getInterpreter() {
    return interpreter;
  }

  @Override
  public String getStubShebang() {
    return stubShebang;
  }

  @Override
  @Nullable
  public Artifact getBootstrapTemplate() {
    return bootstrapTemplate;
  }

  @Nullable
  public NestedSet<Artifact> getFiles() {
    try {
      return files == null ? null : files.getSet(Artifact.class);
    } catch (Depset.TypeException ex) {
      throw new IllegalStateException("for files, " + ex.getMessage());
    }
  }

  @Override
  @Nullable
  public Depset getFilesForStarlark() {
    return files;
  }

  @Override
  @Nullable
  public Artifact getCoverageTool() {
    return coverageTool;
  }

  @Nullable
  public NestedSet<Artifact> getCoverageToolFiles() {
    try {
      return coverageFiles == null ? null : coverageFiles.getSet(Artifact.class);
    } catch (Depset.TypeException ex) {
      throw new IllegalStateException("for coverage_runfiles, " + ex.getMessage());
    }
  }

  @Override
  @Nullable
  public Depset getCoverageToolFilesForStarlark() {
    return coverageFiles;
  }

  public PythonVersion getPythonVersion() {
    return pythonVersion;
  }

  @Override
  public String getPythonVersionForStarlark() {
    return pythonVersion.name();
  }

  /** The class of the {@code PyRuntimeInfo} provider type. */
  public static class PyRuntimeInfoProvider extends BuiltinProvider<PyRuntimeInfo>
      implements PyRuntimeInfoApi.PyRuntimeInfoProviderApi {

    private PyRuntimeInfoProvider() {
      super(STARLARK_NAME, PyRuntimeInfo.class);
    }

    @Override
    public PyRuntimeInfo constructor(
        Object interpreterPathUncast,
        Object interpreterUncast,
        Object filesUncast,
        Object coverageToolUncast,
        Object coverageFilesUncast,
        String pythonVersion,
        String stubShebang,
        Object bootstrapTemplateUncast,
        StarlarkThread thread)
        throws EvalException {
      String interpreterPath =
          interpreterPathUncast == NONE ? null : (String) interpreterPathUncast;
      Artifact interpreter = interpreterUncast == NONE ? null : (Artifact) interpreterUncast;
      Artifact bootstrapTemplate = null;
      if (bootstrapTemplateUncast != NONE) {
        bootstrapTemplate = (Artifact) bootstrapTemplateUncast;
      }
      Depset filesDepset = null;
      if (filesUncast != NONE) {
        // Validate type of filesDepset.
        Depset.cast(filesUncast, Artifact.class, "files");
        filesDepset = (Depset) filesUncast;
      }
      Artifact coverageTool = coverageToolUncast == NONE ? null : (Artifact) coverageToolUncast;
      Depset coverageDepset = null;
      if (coverageFilesUncast != NONE) {
        // Validate type of filesDepset.
        Depset.cast(coverageFilesUncast, Artifact.class, "coverage_files");
        coverageDepset = (Depset) coverageFilesUncast;
      }

      if ((interpreter == null) == (interpreterPath == null)) {
        throw Starlark.errorf(
            "exactly one of the 'interpreter' or 'interpreter_path' arguments must be specified");
      }
      boolean isInBuildRuntime = interpreter != null;
      if (!isInBuildRuntime && filesDepset != null) {
        throw Starlark.errorf("cannot specify 'files' if 'interpreter_path' is given");
      }

      PythonVersion parsedPythonVersion;
      try {
        parsedPythonVersion = PythonVersion.parseTargetValue(pythonVersion);
      } catch (IllegalArgumentException ex) {
        throw Starlark.errorf("illegal value for 'python_version': %s", ex.getMessage());
      }

      Location loc = thread.getCallerLocation();
      if (isInBuildRuntime) {
        if (filesDepset == null) {
          filesDepset = Depset.of(Artifact.class, NestedSetBuilder.emptySet(Order.STABLE_ORDER));
        }
        return new PyRuntimeInfo(
            loc,
            /* interpreterPath= */ null,
            interpreter,
            filesDepset,
            coverageTool,
            coverageDepset,
            parsedPythonVersion,
            stubShebang,
            bootstrapTemplate);
      } else {
        return new PyRuntimeInfo(
            loc,
            PathFragment.create(interpreterPath),
            /* interpreter= */ null,
            /* files= */ null,
            coverageTool,
            coverageDepset,
            parsedPythonVersion,
            stubShebang,
            bootstrapTemplate);
      }
    }
  }
}
