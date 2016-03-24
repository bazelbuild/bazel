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
package com.google.devtools.build.android;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Represents a DataKey for assets.
 *
 * Assets are added to a single directory inside an apk by aapt. Therefore, to determine overwritten
 * and conflicting assets we take the relative from the asset directory and turn it into a DataKey.
 * This serves as the unique identifier for each apk stored asset.
 *
 * Note: Assets have no qualifiers or packages.
 */
public class RelativeAssetPath implements DataKey, Comparable<RelativeAssetPath> {

  /**
   * A Factory that creates RelativeAssetsPath objects whose paths are relative to a given path.
   */
  public static class Factory {
    private final Path assetRoot;

    private Factory(Path assetRoot) {
      this.assetRoot = assetRoot;
    }

    /**
     * Creates a new factory with the asset directory that contains assets.
     */
    public static Factory of(Path assetRoot) {
      return new Factory(Preconditions.checkNotNull(assetRoot));
    }

    public RelativeAssetPath create(Path assetPath) {
      if (!assetPath.startsWith(assetRoot)) {
        throw new IllegalArgumentException(
            String.format("Asset path %s should reside under asset root %s", assetPath, assetRoot));
      }
      return RelativeAssetPath.of(assetRoot.relativize(assetPath));
    }
  }

  private final Path relativeAssetPath;

  private RelativeAssetPath(Path relativeAssetPath) {
    this.relativeAssetPath = relativeAssetPath;
  }

  public static RelativeAssetPath of(Path relativeAssetPath) {
    return new RelativeAssetPath(relativeAssetPath);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (other == null || getClass() != other.getClass()) {
      return false;
    }
    RelativeAssetPath that = (RelativeAssetPath) other;
    return Objects.equals(relativeAssetPath, that.relativeAssetPath);
  }

  @Override
  public int hashCode() {
    return relativeAssetPath.hashCode();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("relativeAssetPath", relativeAssetPath).toString();
  }

  @Override
  public int compareTo(RelativeAssetPath relativeAssetPath) {
    return this.relativeAssetPath.compareTo(relativeAssetPath.relativeAssetPath);
  }
}
