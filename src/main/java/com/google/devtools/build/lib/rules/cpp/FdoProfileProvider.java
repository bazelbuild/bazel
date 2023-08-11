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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.NativeInfo;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Provider that contains the profile used for FDO. */
@Immutable
public final class FdoProfileProvider extends NativeInfo implements StarlarkValue {
  public static final BuiltinProvider<FdoProfileProvider> PROVIDER =
      new BuiltinProvider<FdoProfileProvider>("FdoProfileInfo", FdoProfileProvider.class) {};

  private final FdoInputFile fdoInputFile;
  private final Artifact protoProfileArtifact;

  public FdoProfileProvider(FdoInputFile fdoInputFile, Artifact protoProfileArtifact) {
    this.fdoInputFile = fdoInputFile;
    this.protoProfileArtifact = protoProfileArtifact;
  }

  @Override
  public BuiltinProvider<FdoProfileProvider> getProvider() {
    return PROVIDER;
  }

  @StarlarkMethod(
      name = "get_fdo_artifact",
      documented = false,
      allowReturnNones = true,
      useStarlarkThread = true)
  @Nullable
  public Artifact getFdoArtifact(StarlarkThread thread) throws EvalException {
    BuiltinRestriction.failIfCalledOutsideBuiltins(thread);
    return fdoInputFile.getArtifact();
  }

  public FdoInputFile getInputFile() {
    return fdoInputFile;
  }

  public Artifact getProtoProfileArtifact() {
    return protoProfileArtifact;
  }
}
