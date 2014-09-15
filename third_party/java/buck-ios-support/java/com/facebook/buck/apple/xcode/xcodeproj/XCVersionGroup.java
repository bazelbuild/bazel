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

import com.facebook.buck.apple.xcode.XcodeprojSerializer;
import com.google.common.base.Optional;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.Lists;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class XCVersionGroup extends PBXReference {
  private Optional<PBXFileReference> currentVersion;

  private final List<PBXFileReference> children;

  private final LoadingCache<SourceTreePath, PBXFileReference> fileReferencesBySourceTreePath;


  public XCVersionGroup(String name, String path, SourceTree sourceTree) {
    super(name, path, sourceTree);
    children = Lists.newArrayList();

    fileReferencesBySourceTreePath = CacheBuilder.newBuilder().build(
        new CacheLoader<SourceTreePath, PBXFileReference>() {
          @Override
          public PBXFileReference load(SourceTreePath key) throws Exception {
            PBXFileReference ref = new PBXFileReference(
                key.getPath().getFileName().toString(),
                key.getPath().toString(),
                key.getSourceTree());
            children.add(ref);
            return ref;
          }
        });

    currentVersion = Optional.absent();
  }

  public Optional<String> getVersionGroupType() {
    if (currentVersion.isPresent()) {
      return currentVersion.get().getExplicitFileType();
    }
    return Optional.absent();
  }

  public Optional<PBXFileReference> getCurrentVersion() { return currentVersion; }

  public void setCurrentVersion(Optional<PBXFileReference> v) { currentVersion = v; }

  public List<PBXFileReference> getChildren() {
    return children;
  }

  public PBXFileReference getOrCreateFileReferenceBySourceTreePath(SourceTreePath sourceTreePath) {
    return fileReferencesBySourceTreePath.getUnchecked(sourceTreePath);
  }

  @Override
  public String isa() {
    return "XCVersionGroup";
  }

  @Override
  public void serializeInto(XcodeprojSerializer s) {
    super.serializeInto(s);

    Collections.sort(children, new Comparator<PBXReference>() {
          @Override
          public int compare(PBXReference o1, PBXReference o2) {
          return o1.getName().compareTo(o2.getName());
        }
      });
    s.addField("children", children);


    if (currentVersion.isPresent()) {
      s.addField("currentVersion", currentVersion.get());
    }

    Optional<String> versionGroupType = getVersionGroupType();
    if (versionGroupType.isPresent()) {
      s.addField("versionGroupType", versionGroupType.get());
    }
  }
}
