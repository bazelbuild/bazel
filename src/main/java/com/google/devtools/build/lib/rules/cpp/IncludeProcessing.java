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

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScannerSupplier;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import java.util.List;
import javax.annotation.Nullable;

/** Used as an interface to thin header inputs to compile actions for C++-like compiles. */
public interface IncludeProcessing {
  /** Performs include processing actions and returns the processed set of resulting headers. */
  List<Artifact> determineAdditionalInputs(
      @Nullable IncludeScannerSupplier includeScannerSupplier,
      CppCompileAction action,
      ActionExecutionContext actionExecutionContext,
      IncludeScanningHeaderData includeScanningHeaderData)
      throws ExecException, InterruptedException;
}
