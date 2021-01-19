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

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CppIncludeScanningContext;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScannerSupplier;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import com.google.devtools.build.lib.rules.cpp.IncludeScanning;
import java.util.List;

/**
 * Include scanning context implementation.
 */
public class CppIncludeScanningContextImpl implements CppIncludeScanningContext {

  private final Supplier<? extends IncludeScannerSupplier> includeScannerSupplier;

  public CppIncludeScanningContextImpl(
      Supplier<? extends IncludeScannerSupplier> includeScannerSupplier) {
    this.includeScannerSupplier = includeScannerSupplier;
  }

  @Override
  public List<Artifact> findAdditionalInputs(
      CppCompileAction action,
      ActionExecutionContext actionExecutionContext,
      IncludeScanningHeaderData includeScanningHeaderData)
      throws ExecException, InterruptedException {
    return IncludeScanning.INSTANCE.determineAdditionalInputs(
        includeScannerSupplier.get(), action, actionExecutionContext, includeScanningHeaderData);
  }
}
