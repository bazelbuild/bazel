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

import static com.google.devtools.build.lib.syntax.Runtime.NONE;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.skylarkbuildapi.python.PyRuntimeInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import javax.annotation.Nullable;

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
public class PyRuntimeInfo extends Info implements PyRuntimeInfoApi<Artifact> {

  /** The Starlark-accessible top-level builtin name for this provider type. */
  public static final String STARLARK_NAME = "PyRuntimeInfo";

  /** The singular {@code PyRuntimeInfo} provider type object. */
  public static final PyRuntimeInfoProvider PROVIDER = new PyRuntimeInfoProvider();

  @Nullable private final PathFragment interpreterPath;
  @Nullable private final Artifact interpreter;
  @Nullable private final SkylarkNestedSet files;
  /** Invariant: either PY2 or PY3. */
  private final PythonVersion pythonVersion;

  private PyRuntimeInfo(
      @Nullable Location location,
      @Nullable PathFragment interpreterPath,
      @Nullable Artifact interpreter,
      @Nullable SkylarkNestedSet files,
      PythonVersion pythonVersion) {
    super(PROVIDER, location);
    Preconditions.checkArgument((interpreterPath == null) != (interpreter == null));
    Preconditions.checkArgument((interpreter == null) == (files == null));
    Preconditions.checkArgument(pythonVersion.isTargetValue());
    if (files != null) {
      // Work around #7266 by special-casing the empty set in the type check.
      Preconditions.checkArgument(
          files.isEmpty() || files.getContentType().canBeCastTo(Artifact.class));
    }
    this.interpreterPath = interpreterPath;
    this.interpreter = interpreter;
    this.files = files;
    this.pythonVersion = pythonVersion;
  }

  /** Constructs an instance from native rule logic (built-in location) for an in-build runtime. */
  public static PyRuntimeInfo createForInBuildRuntime(
      Artifact interpreter, NestedSet<Artifact> files, PythonVersion pythonVersion) {
    return new PyRuntimeInfo(
        /*location=*/ null,
        /*interpreterPath=*/ null,
        interpreter,
        SkylarkNestedSet.of(Artifact.class, files),
        pythonVersion);
  }

  /** Constructs an instance from native rule logic (built-in location) for a platform runtime. */
  public static PyRuntimeInfo createForPlatformRuntime(
      PathFragment interpreterPath, PythonVersion pythonVersion) {
    return new PyRuntimeInfo(
        /*location=*/ null, interpreterPath, /*interpreter=*/ null, /*files=*/ null, pythonVersion);
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
        && this.files.equals(otherInfo.files));
  }

  @Override
  public int hashCode() {
    return Objects.hash(PyRuntimeInfo.class, interpreterPath, interpreter, files);
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

  @Nullable
  public NestedSet<Artifact> getFiles() {
    return files == null ? null : files.getSet(Artifact.class);
  }

  @Override
  @Nullable
  public SkylarkNestedSet getFilesForStarlark() {
    return files;
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
        String pythonVersion,
        Location loc)
        throws EvalException {
      String interpreterPath =
          interpreterPathUncast == NONE ? null : (String) interpreterPathUncast;
      Artifact interpreter = interpreterUncast == NONE ? null : (Artifact) interpreterUncast;
      SkylarkNestedSet files = filesUncast == NONE ? null : (SkylarkNestedSet) filesUncast;

      if ((interpreter == null) == (interpreterPath == null)) {
        throw new EvalException(
            loc,
            "exactly one of the 'interpreter' or 'interpreter_path' arguments must be specified");
      }
      boolean isInBuildRuntime = interpreter != null;
      if (!isInBuildRuntime && files != null) {
        throw new EvalException(loc, "cannot specify 'files' if 'interpreter_path' is given");
      }

      PythonVersion parsedPythonVersion;
      try {
        parsedPythonVersion = PythonVersion.parseTargetValue(pythonVersion);
      } catch (IllegalArgumentException ex) {
        throw new EvalException(loc, "illegal value for 'python_version'", ex);
      }

      if (isInBuildRuntime) {
        if (files == null) {
          files =
              SkylarkNestedSet.of(Artifact.class, NestedSetBuilder.emptySet(Order.STABLE_ORDER));
        }
        return new PyRuntimeInfo(
            loc, /*interpreterPath=*/ null, interpreter, files, parsedPythonVersion);
      } else {
        return new PyRuntimeInfo(
            loc,
            PathFragment.create(interpreterPath),
            /*interpreter=*/ null,
            /*files=*/ null,
            parsedPythonVersion);
      }
    }
  }
}
