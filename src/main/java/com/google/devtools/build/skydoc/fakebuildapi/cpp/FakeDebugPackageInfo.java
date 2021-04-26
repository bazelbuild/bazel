// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.fakebuildapi.cpp;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.DebugPackageInfoApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;

/** Fake implementation of {@link DebugPackageInfoApi}. */
public class FakeDebugPackageInfo implements DebugPackageInfoApi<FileApi> {

  @Override
  public Label getTargetLabel() {
    return null;
  }

  @Override
  public FileApi getStrippedArtifact() {
    return null;
  }

  @Override
  public FileApi getUnstrippedArtifact() {
    return null;
  }

  /** Returns the .dwp file (for fission builds) or null if --fission=no. */
  @Override
  public final FileApi getDwpArtifact() {
    return null;
  }

  @Override
  public String toProto() throws EvalException {
    return null;
  }

  @Override
  public String toJson() throws EvalException {
    return null;
  }

  @Override
  public void repr(Printer printer) {}

  /** Fake implementation of {@link DebugPackageInfoApi.Provider}. */
  public static class Provider implements DebugPackageInfoApi.Provider<FileApi> {

    @Override
    public DebugPackageInfoApi<FileApi> createDebugPackageInfo(
        Label targetLabel, Object strippedArtifact, FileApi unstrippedArtifact, Object dwpArtifact)
        throws EvalException {
      return new FakeDebugPackageInfo();
    }

    @Override
    public void repr(Printer printer) {}
  }
}
