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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScannerSupplier;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningPreparer;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;

/** Runs include scanning on actions, if include scanning is enabled. */
public class IncludeScanning implements IncludeProcessing {
  @AutoCodec public static final IncludeScanning INSTANCE = new IncludeScanning();

  @Override
  public @Nullable Iterable<Artifact> determineAdditionalInputs(
      @Nullable IncludeScannerSupplier includeScannerSupplier,
      CppCompileAction action,
      ActionExecutionContext actionExecutionContext,
      IncludeScanningHeaderData includeScanningHeaderData)
      throws ExecException, InterruptedException {
    // Return null when no include scanning occurs, as opposed to an empty set, to distinguish from
    // the case where includes are scanned but none are found.
    if (!action.shouldScanIncludes()) {
      return null;
    }

    return IncludeScanningPreparer.scanForIncludedInputs(
        action,
        Preconditions.checkNotNull(includeScannerSupplier, action),
        includeScanningHeaderData,
        action,
        actionExecutionContext,
        action.getSourceFile().getExecPathString());
  }
}
