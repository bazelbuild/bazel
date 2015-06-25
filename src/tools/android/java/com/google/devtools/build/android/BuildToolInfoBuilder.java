// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.android;

import com.android.SdkConstants;
import com.android.sdklib.BuildToolInfo;
import com.android.sdklib.repository.FullRevision;

import java.io.File;
import java.nio.file.Path;

import javax.annotation.Nullable;

/**
 * Simplifies the creation of a {@link BuildToolInfo}.
 */
public class BuildToolInfoBuilder {
  private File aaptLocation;
  private FullRevision fullRevision;
  private File zipAlign;

  public BuildToolInfoBuilder(FullRevision fullRevision) {
    this.fullRevision = fullRevision;
  }

  public BuildToolInfoBuilder setAapt(@Nullable Path aaptLocation) {
    this.aaptLocation = aaptLocation != null ? aaptLocation.toFile() : null;
    return this;
  }

  public BuildToolInfoBuilder setZipAlign(@Nullable Path zipAlign) {
    this.zipAlign = zipAlign != null ? zipAlign.toFile() : null;
    return this;
  }

  public BuildToolInfo build() {
    // Fill in the unused tools with fakes that will make sense if unexpectedly called.
    Path platformToolsRoot = new File("unused/path/to/sdk/root").toPath();
    return new BuildToolInfo(fullRevision,
        platformToolsRoot.toFile(),
        aaptLocation,
        platformToolsRoot.resolve(SdkConstants.FN_AIDL).toFile(),
        platformToolsRoot.resolve(SdkConstants.FN_DX).toFile(),
        platformToolsRoot.resolve(SdkConstants.FN_DX_JAR).toFile(),
        platformToolsRoot.resolve(SdkConstants.FN_RENDERSCRIPT).toFile(),
        platformToolsRoot.resolve(SdkConstants.FN_FRAMEWORK_INCLUDE).toFile(),
        platformToolsRoot.resolve(SdkConstants.FN_FRAMEWORK_INCLUDE_CLANG).toFile(),
        platformToolsRoot.resolve(SdkConstants.FN_BCC_COMPAT).toFile(),
        platformToolsRoot.resolve(SdkConstants.FN_LD_ARM).toFile(),
        platformToolsRoot.resolve(SdkConstants.FN_LD_X86).toFile(),
        platformToolsRoot.resolve(SdkConstants.FN_LD_MIPS).toFile(),
        zipAlign == null ? platformToolsRoot.resolve(SdkConstants.FN_ZIPALIGN).toFile() : zipAlign);
  }
}
