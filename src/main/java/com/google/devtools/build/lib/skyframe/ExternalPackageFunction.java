// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * A SkyFunction for parsing the {@code //external} package.
 *
 * <p>This function iterates through the WorkspaceFileValue-s to get the last WorkspaceFileValue
 * that will contain all the bind statements from the WORKSPACE file.
 */
public class ExternalPackageFunction implements SkyFunction {
  @AutoCodec @AutoCodec.VisibleForSerialization
  static final SkyKey KEY = () -> SkyFunctions.EXTERNAL_PACKAGE;

  private final ExternalPackageHelper externalPackageHelper;

  public ExternalPackageFunction(ExternalPackageHelper externalPackageHelper) {
    this.externalPackageHelper = externalPackageHelper;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    RootedPath workspacePath = externalPackageHelper.findWorkspaceFile(env);
    if (env.valuesMissing()) {
      return null;
    }

    // This currently cannot be null due to a hack in ExternalPackageUtil.findWorkspaceFile()
    // TODO(lberki): Remove that hack and handle the case when the WORKSPACE file is not found.
    SkyKey key = WorkspaceFileValue.key(workspacePath);
    WorkspaceFileValue value = (WorkspaceFileValue) env.getValue(key);
    if (value == null) {
      return null;
    }
    // Walk to the last WorkspaceFileValue that accumulate all the bindings of the WORKSPACE
    // file.
    while (value.next() != null) {
      value = (WorkspaceFileValue) env.getValue(value.next());
      if (value == null) {
        return null;
      }
    }
    return new PackageValue(value.getPackage());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /** Returns the singleton {@link SkyKey} for the external package. */
  public static SkyKey key() {
    return KEY;
  }
}
