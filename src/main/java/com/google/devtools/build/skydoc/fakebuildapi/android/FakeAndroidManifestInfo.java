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

package com.google.devtools.build.skydoc.fakebuildapi.android;

import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidManifestInfoApi;
import net.starlark.java.eval.EvalException;

/** Fake implementation of AndroidManifestInfo. */
public class FakeAndroidManifestInfo implements AndroidManifestInfoApi<FileApi> {

  @Override
  public FileApi getManifest() {
    return null;
  }

  @Override
  public String getPackage() {
    return null;
  }

  @Override
  public boolean exportsManifest() {
    return false;
  }

  @Override
  public String toProto() throws EvalException {
    return null;
  }

  @Override
  public String toJson() throws EvalException {
    return null;
  }

  /** Fake implementation of {@link AndroidManifestInfoApi.Provider}. */
  public static class FakeProvider implements AndroidManifestInfoApi.Provider<FileApi> {

    @Override
    public AndroidManifestInfoApi<FileApi> androidManifestInfo(
        FileApi manifest, String packageString, Boolean exportsManifest) throws EvalException {
      return new FakeAndroidManifestInfo();
    }
  }
}
