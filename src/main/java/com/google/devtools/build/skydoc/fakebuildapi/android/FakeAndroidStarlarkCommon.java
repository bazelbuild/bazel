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

package com.google.devtools.build.skydoc.fakebuildapi.android;

import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidDeviceBrokerInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidSplitTransititionApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidStarlarkCommonApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaOutputApi;

/** Fake implementation of {@link AndroidStarlarkCommonApi}. */
public class FakeAndroidStarlarkCommon
    implements AndroidStarlarkCommonApi<FileApi, JavaInfoApi<FileApi, JavaOutputApi<FileApi>>> {

  @Override
  public AndroidDeviceBrokerInfoApi createDeviceBrokerInfo(String deviceBrokerType) {
    return new FakeAndroidDeviceBrokerInfo();
  }

  @Override
  public String getSourceDirectoryRelativePathFromResource(FileApi resource) {
    return null;
  }

  @Override
  public AndroidSplitTransititionApi getAndroidSplitTransition() {
    return new FakeAndroidSplitTransitition();
  }

  @Override
  public JavaInfoApi<FileApi, JavaOutputApi<FileApi>>
      enableImplicitSourcelessDepsExportsCompatibility(
          JavaInfoApi<FileApi, JavaOutputApi<FileApi>> javaInfo) {
    return null;
  }
}
