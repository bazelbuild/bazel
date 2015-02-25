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

import com.facebook.buck.apple.xcode.XcodeprojSerializer;
import com.facebook.buck.apple.xcode.xcodeproj.PBXTargetDependency;

/**
 * A target dependency that is not remote, which is similar to the normal Buck PBXTargetDependency,
 * but includes a {@code target} field.
 * <p>
 * TODO(bazel-team): Upstream this to Buck.
 */
public class LocalPBXTargetDependency extends PBXTargetDependency {
  public LocalPBXTargetDependency(LocalPBXContainerItemProxy targetProxy) {
    super(targetProxy);
  }

  @Override
  public void serializeInto(XcodeprojSerializer s) {
    super.serializeInto(s);
    s.addField("target", getTargetProxy().getRemoteGlobalIDString());
  }
}
