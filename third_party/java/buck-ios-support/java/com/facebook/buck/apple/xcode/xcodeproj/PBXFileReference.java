/*
 * Copyright 2013-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package com.facebook.buck.apple.xcode.xcodeproj;

import com.facebook.buck.apple.FileTypes;
import com.facebook.buck.apple.xcode.XcodeprojSerializer;
import com.google.common.base.Optional;
import com.google.common.io.Files;

/**
 * Reference to a concrete file.
 */
public class PBXFileReference extends PBXReference {
  private Optional<String> explicitFileType;
  private Optional<String> lastKnownFileType;

  public PBXFileReference(String name, String path, SourceTree sourceTree) {
    super(name, path, sourceTree);

    // this is necessary to prevent O(n^2) behavior in xcode project loading
    String fileType = FileTypes.FILE_EXTENSION_TO_UTI.get(Files.getFileExtension(name));
    if (FileTypes.EXPLICIT_FILE_TYPE_BROKEN_UTIS.contains(fileType)) {
      explicitFileType = Optional.absent();
      lastKnownFileType = Optional.of(fileType);
    } else {
      explicitFileType = Optional.fromNullable(fileType);
      lastKnownFileType = Optional.absent();
    }
  }

  public Optional<String> getExplicitFileType() {
    return explicitFileType;
  }

  public void setExplicitFileType(Optional<String> explicitFileType) {
    this.explicitFileType = explicitFileType;
  }

  @Override
  public String isa() {
    return "PBXFileReference";
  }

  @Override
  public void serializeInto(XcodeprojSerializer s) {
    super.serializeInto(s);

    if (explicitFileType.isPresent()) {
      s.addField("explicitFileType", explicitFileType.get());
    }

    if (lastKnownFileType.isPresent()) {
      s.addField("lastKnownFileType", lastKnownFileType.get());
    }
  }

  @Override
  public String toString() {
    return String.format(
        "%s explicitFileType=%s",
        super.toString(),
        getExplicitFileType());
  }
}
