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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraLinkArgs;

/**
 * Implementation for the "ios_extension_binary" rule.
 */
public class IosExtensionBinary extends BinaryLinkingTargetFactory {
  public IosExtensionBinary() {
    super(HasReleaseBundlingSupport.NO,
        new ExtraLinkArgs("-e", "_NSExtensionMain", "-fapplication-extension"),
        XcodeProductType.LIBRARY_STATIC);
  }
}
