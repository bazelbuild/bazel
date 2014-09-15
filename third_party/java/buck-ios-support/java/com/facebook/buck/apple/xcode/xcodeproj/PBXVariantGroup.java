/*
 * Copyright 2014-present Facebook, Inc.
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

import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;

import java.util.Objects;

public class PBXVariantGroup extends PBXGroup {

  private final LoadingCache<VirtualNameAndSourceTreePath, PBXFileReference>
      variantFileReferencesByNameAndSourceTreePath;

  public PBXVariantGroup(String name, String path, SourceTree sourceTree) {
    super(name, path, sourceTree);

    variantFileReferencesByNameAndSourceTreePath = CacheBuilder.newBuilder().build(
        new CacheLoader<VirtualNameAndSourceTreePath, PBXFileReference>() {
          @Override
          public PBXFileReference load(VirtualNameAndSourceTreePath key) throws Exception {
            PBXFileReference ref = new PBXFileReference(
                key.getVirtualName(),
                key.getSourceTreePath().getPath().toString(),
                key.getSourceTreePath().getSourceTree());
            getChildren().add(ref);
            return ref;
          }
        });
  }

  public PBXFileReference getOrCreateVariantFileReferenceByNameAndSourceTreePath(
      String virtualName,
      SourceTreePath sourceTreePath) {
    VirtualNameAndSourceTreePath key =
        new VirtualNameAndSourceTreePath(virtualName, sourceTreePath);
    return variantFileReferencesByNameAndSourceTreePath.getUnchecked(key);
  }

  @Override
  public String isa() {
    return "PBXVariantGroup";
  }

  private static class VirtualNameAndSourceTreePath {
    private final String virtualName;
    private final SourceTreePath sourceTreePath;

    public VirtualNameAndSourceTreePath(String virtualName, SourceTreePath sourceTreePath) {
      this.virtualName = Preconditions.checkNotNull(virtualName);
      this.sourceTreePath = Preconditions.checkNotNull(sourceTreePath);
    }

    public String getVirtualName() {
      return virtualName;
    }

    public SourceTreePath getSourceTreePath() {
      return sourceTreePath;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof VirtualNameAndSourceTreePath)) {
        return false;
      }

      VirtualNameAndSourceTreePath that = (VirtualNameAndSourceTreePath) other;
      return Objects.equals(this.virtualName, that.virtualName) &&
          Objects.equals(this.sourceTreePath, that.sourceTreePath);
    }

    @Override
    public int hashCode() {
      return Objects.hash(virtualName, sourceTreePath);
    }
  }
}
