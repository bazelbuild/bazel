// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.xcode.xcodegen;

import com.google.common.base.Preconditions;

import com.facebook.buck.apple.xcode.XcodeprojSerializer;
import com.facebook.buck.apple.xcode.xcodeproj.PBXContainerItemProxy;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXObject;
import com.facebook.buck.apple.xcode.xcodeproj.PBXProject;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;

/**
 * Represents a PBXContainerItemProxy object that does not reference a remote (other project file)
 * object.
 * <p>
 * TODO(bazel-team): Upstream this to Buck.
 */
public class LocalPBXContainerItemProxy extends PBXContainerItemProxy {
  private static final PBXFileReference DUMMY_FILE_REFERENCE =
      new PBXFileReference("", "", SourceTree.ABSOLUTE);

  private final PBXProject containerPortalAsProject;
  private final PBXObject remoteGlobalIdHolder;

  public LocalPBXContainerItemProxy(
      PBXProject containerPortalAsProject, PBXObject remoteGlobalIdHolder, ProxyType proxyType) {
    super(DUMMY_FILE_REFERENCE, "" /* remoteGlobalIDString */, proxyType);
    this.containerPortalAsProject = Preconditions.checkNotNull(containerPortalAsProject);
    this.remoteGlobalIdHolder = Preconditions.checkNotNull(remoteGlobalIdHolder);
  }

  public PBXProject getContainerPortalAsProject() {
    return containerPortalAsProject;
  }

  public PBXObject getRemoteGlobalIdHolder() {
    return remoteGlobalIdHolder;
  }

  @Override
  public void serializeInto(XcodeprojSerializer s) {
    // Note we don't call super.serializeInto because we don't want the DUMMY_FILE_REFERENCE to
    // get saved to the project file (even though it is probably harmless).
    s.addField("containerPortal", containerPortalAsProject.getGlobalID());
    s.addField("remoteGlobalIDString", remoteGlobalIdHolder);
    s.addField("proxyType", getProxyType().getIntValue());
  }
}
