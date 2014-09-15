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

import com.google.devtools.build.xcode.util.Mapping;

import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;

import java.util.HashMap;
import java.util.Map;

/**
 * A factory for {@link PBXFileReference}s. {@link PBXFileReference}s are inherently non-value
 * types, in that each created instance appears in the serialized Xcodeproj file, even if some of
 * the instances are equivalent. This serves as a cache so that a value type ({@link FileReference})
 * can be converted to a canonical, cached {@link PBXFileReference} which is equivalent to it.
 */
final class PBXFileReferences {
  private final Map<FileReference, PBXFileReference> cache = new HashMap<>();

  /**
   * Supplies a reference, containing values for fields specified in {@code reference}.
   */
  public PBXFileReference get(FileReference reference) {
    for (PBXFileReference existing : Mapping.of(cache, reference).asSet()) {
      return existing;
    }

    PBXFileReference result = new PBXFileReference(
        reference.name(), reference.path().orNull(), reference.sourceTree());
    result.setExplicitFileType(reference.explicitFileType());

    cache.put(reference, result);
    return result;
  }
}
