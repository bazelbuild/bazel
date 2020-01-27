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

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * A SkyFunction for resolving //external:* bindings.
 *
 * <p>This function iterates through the WorkspaceFileValue-s to get the last WorkspaceFileValue
 * that will contains all the bind statements from the workspace file.
 */
public class ExternalPackageFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    RootedPath workspacePath = (RootedPath) skyKey.argument();
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

  /** Returns a {@link Key} to find the WORKSPACE file at the given path. */
  public static SkyKey key(RootedPath workspacePath) {
    return Key.create(workspacePath);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<RootedPath> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(RootedPath arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(RootedPath arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.EXTERNAL_PACKAGE;
    }
  }
}
