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

import com.dd.plist.NSDictionary;
import com.facebook.buck.apple.xcode.XcodeprojSerializer;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;

/**
 * File referenced by a build phase, unique to each build phase.
 *
 * Contains a dictionary {@link #settings} which holds additional information to be interpreted by
 * the particular phase referencing this object, e.g.:
 *
 *  - {@link PBXHeadersBuildPhase } may read <code>{"ATTRIBUTES": ["Public"]}</code> and interpret
 *    the build file as a public (exported) header.
 *  - {@link PBXSourcesBuildPhase } may read <code>{"COMPILER_FLAGS": "-foo"}</code> and interpret
 *    that this file should be compiled with the additional flag {@code "-foo" }.
 */
public class PBXBuildFile extends PBXProjectItem {
  private final PBXReference fileRef;
  private Optional<NSDictionary> settings;

  public PBXBuildFile(PBXReference fileRef) {
    this.fileRef = Preconditions.checkNotNull(fileRef);
    this.settings = Optional.absent();
  }

  public PBXReference getFileRef() {
    return fileRef;
  }

  public Optional<NSDictionary> getSettings() {
    return settings;
  }

  public void setSettings(Optional<NSDictionary> v) {
    settings = v;
  }

  @Override
  public String isa() {
    return "PBXBuildFile";
  }

  @Override
  public int stableHash() {
    return fileRef.stableHash();
  }

  @Override
  public void serializeInto(XcodeprojSerializer s) {
    super.serializeInto(s);

    s.addField("fileRef", fileRef);
    if (settings.isPresent()) {
      s.addField("settings", settings.get());
    }
  }
}
