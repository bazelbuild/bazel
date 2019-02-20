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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** Information about the Python runtime used by the <code>py_*</code> rules. */
@AutoValue
@Immutable
public abstract class PyRuntimeProvider implements TransitiveInfoProvider {

  public static PyRuntimeProvider create(
      @Nullable NestedSet<Artifact> files,
      @Nullable Artifact interpreter,
      @Nullable PathFragment interpreterPath) {
    return new AutoValue_PyRuntimeProvider(files, interpreter, interpreterPath);
  }

  /**
   * Returns whether this runtime is hermetic, i.e. represents an in-build interpreter as opposed to
   * a system interpreter.
   *
   * <p>Hermetic runtimes have non-null values for {@link #getInterpreter} and {@link #getFiles},
   * while non-hermetic runtimes have non-null {@link #getInterpreterPath}.
   *
   * <p>Note: Despite the name, it is still possible for a hermetic runtime to reference in-build
   * files that have non-hermetic behavior. For example, {@link #getInterpreter} could reference a
   * checked-in wrapper script that calls the system interpreter at execution time.
   */
  public boolean isHermetic() {
    return getInterpreter() != null;
  }

  @Nullable
  public abstract NestedSet<Artifact> getFiles();

  @Nullable
  public abstract Artifact getInterpreter();

  @Nullable
  public abstract PathFragment getInterpreterPath();
}
