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

import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;

/**
 * Implements a strategy to group self-contained PBXReferences into PBXGroups. Grouping is done to
 * make it easier to navigate the files of the project in Xcode's Project Navigator.
 * <p>
 * A <em>self-contained</em> reference is one that is not a member of a PBXVariantGroup or other
 * aggregate group, although a self-contained reference may contain such a reference as a child.
 */
public interface Grouper {
  /**
   * Takes self-contained references that could be placed in the main group, and groups them into
   * one or more references that can also be placed in the main group.
   * <p>
   * The on-disk path of the main group is assumed to point to workspace root. This is important
   * when the {@code sourceRoot} property of the references in the input or output arguments are
   * {@link SourceTree#GROUP}.
   */
  Iterable<PBXReference> group(Iterable<PBXReference> references);
}
