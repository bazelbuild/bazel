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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionContextMarker;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Set;

/**
 * Context for actions that do swig include scanning.
 */
@ActionContextMarker(name = "SwigIncludeScanning")
public interface SwigIncludeScanningContext extends ActionContext {
  /**
   * Scans includes in a .swig file ({@code source}), placing the results into {@code includes}.
   *
   * <p>Like {@link IncludeScanner#processAsync}, may short-circuit if a skyframe restart is
   * necessary. Callers must check {@link
   * com.google.devtools.build.skyframe.SkyFunction.Environment#valuesMissing}.
   */
  void extractSwigIncludes(
      Set<Artifact> includes,
      ActionExecutionMetadata actionExecutionMetadata,
      ActionExecutionContext actionExecContext,
      Artifact source,
      ImmutableSet<Artifact> legalOutputPaths,
      ImmutableList<PathFragment> swigIncludePaths,
      Artifact grepIncludes)
      throws IOException, ExecException, InterruptedException;
}
