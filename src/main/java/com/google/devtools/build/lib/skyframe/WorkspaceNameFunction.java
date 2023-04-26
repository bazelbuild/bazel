// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * {@link SkyFunction} for {@link WorkspaceNameValue}s.
 *
 * <p>All errors (e.g. parsing errors or a symlink cycle encountered when consuming the WORKSPACE
 * file) result in a {@link com.google.devtools.build.lib.packages.NoSuchPackageException}.
 */
public class WorkspaceNameFunction implements SkyFunction {
  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, WorkspaceNameFunctionException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }
    if (starlarkSemantics.getBool(BuildLanguageOptions.ENABLE_BZLMOD)) {
      // When Bzlmod is enabled, we don't care what the "workspace name" specified in the WORKSPACE
      // file is, and always use the static string "_main" instead. The workspace name returned by
      // this SkyFunction is only used as the runfiles/execpath prefix for the main repo; for other
      // repos, the canonical repo name is used. The canonical name of the main repo is the empty
      // string, so we can't use that; instead, we just use a static string.
      //
      // "_main" was chosen because it's not a valid apparent repo name, which, coupled with the
      // fact that no Bzlmod-generated canonical repo names are valid apparent repo names, means
      // that a path passed to rlocation can go through repo mapping multiple times without any
      // danger (i.e. going through repo mapping is idempotent).
      return WorkspaceNameValue.withName("_main");
    }
    PackageValue externalPackageValue =
        (PackageValue) env.getValue(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER);
    if (externalPackageValue == null) {
      return null;
    }
    Package externalPackage = externalPackageValue.getPackage();
    if (externalPackage.containsErrors()) {
      throw new WorkspaceNameFunctionException();
    }
    return WorkspaceNameValue.withName(externalPackage.getWorkspaceName());
  }

  private static class WorkspaceNameFunctionException extends SkyFunctionException {
    WorkspaceNameFunctionException() {
      super(
          new BuildFileContainsErrorsException(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER),
          Transience.PERSISTENT);
    }
  }
}
