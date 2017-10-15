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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.cpp.ArtifactCategory;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.vfs.FileSystemUtils;

/**
 * A {@link CcLinkParamsStore} to be propagated to dependent cc_{library, binary} targets.
 */
class ObjcLibraryCcLinkParamsStore extends CcLinkParamsStore {

  private final ObjcCommon common;

  /**
   * Create a params store.
   * 
   * @param common the {@link ObjcCommon} instance for this target.
   */
  public ObjcLibraryCcLinkParamsStore(ObjcCommon common) {
    this.common = common;
  }

  @Override
  protected void collect(
      CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {
    ObjcProvider objcProvider = common.getObjcProvider();

    ImmutableSet.Builder<LibraryToLink> libraries = new ImmutableSet.Builder<>();
    for (Artifact library : objcProvider.get(ObjcProvider.LIBRARY)) {
      libraries.add(LinkerInputs.opaqueLibraryToLink(
          library, ArtifactCategory.STATIC_LIBRARY,
          FileSystemUtils.removeExtension(library.getRootRelativePathString())));
    }
    libraries.addAll(objcProvider.get(ObjcProvider.CC_LIBRARY));
    builder.addLibraries(libraries.build());

    for (SdkFramework sdkFramework : objcProvider.get(ObjcProvider.SDK_FRAMEWORK)) {
      builder.addLinkOpts(ImmutableList.of("-framework", sdkFramework.getName()));
    }
  }
}