// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.android.builder.dependency.SymbolFileProvider;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import java.io.File;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Contains the assets, resources, manifest and resource symbols for an android_library dependency.
 *
 * <p>This class serves the role of both a processed MergedAndroidData and a dependency exported
 * from another invocation of the AndroidResourcesProcessorAction. Since it's presumed to be cheaper
 * to only pass the derived artifact (rTxt) rather that the entirety of the processed dependencies
 * (png crunching and resource processing should be saved for the final
 * AndroidResourcesProcessorAction invocation) AndroidData can have multiple roots for resources and
 * assets.
 */
class DependencyAndroidData extends SerializedAndroidData {
  private static final Pattern VALID_REGEX = Pattern.compile(".*:.*:.+:.+(:.*){0,2}");

  public static final String EXPECTED_FORMAT =
      "resources[#resources]:assets[#assets]:manifest:r.txt:symbols.bin:static.library.ap_";

  public static DependencyAndroidData valueOf(String text) {
    return valueOf(text, FileSystems.getDefault());
  }

  @VisibleForTesting
  static DependencyAndroidData valueOf(String text, FileSystem fileSystem) {
    if (!VALID_REGEX.matcher(text).find()) {
      throw new IllegalArgumentException(text + " is not in the format '" + EXPECTED_FORMAT + "'");
    }
    String[] parts = text.split(":");
    // TODO(bazel-team): Handle the symbols.bin file.
    // The local symbols.bin is optional -- if it is missing, we'll use the full R.txt
    Path rTxt = exists(fileSystem.getPath(parts[3]));
    ImmutableList<Path> assetDirs =
        parts[1].length() == 0 ? ImmutableList.<Path>of() : splitPaths(parts[1], fileSystem);
    return new DependencyAndroidData(
        splitPaths(parts[0], fileSystem),
        assetDirs,
        exists(fileSystem.getPath(parts[2])),
        rTxt,
        parts.length > 4 ? fileSystem.getPath(parts[4]) : null,
        parts.length > 5
            ? StaticLibrary.from(exists(fileSystem.getPath(parts[5])), rTxt, assetDirs)
            : null);
  }

  private final Path manifest;
  private final Path rTxt;
  private final StaticLibrary staticLibrary;

  public DependencyAndroidData(
      ImmutableList<Path> resourceDirs,
      ImmutableList<Path> assetDirs,
      Path manifest,
      Path rTxt,
      Path symbols,
      StaticLibrary staticLibrary) {
    // Use the manifest as a label for now.
    super(resourceDirs, assetDirs, manifest.toString(), symbols);
    this.manifest = manifest;
    this.rTxt = rTxt;
    this.staticLibrary = staticLibrary;
  }

  public SymbolFileProvider asSymbolFileProvider() {
    return new SymbolFileProvider() {
      @Override
      public File getManifest() {
        return manifest.toFile();
      }

      @Override
      public File getSymbolFile() {
        return rTxt == null ? null : rTxt.toFile();
      }

      @Override
      public boolean isOptional() {
        return false;
      }

      @Override
      public int hashCode() {
        return Objects.hash(getManifest(), getSymbolFile());
      }

      @Override
      public boolean equals(Object obj) {
        if (obj instanceof SymbolFileProvider) {
          SymbolFileProvider other = (SymbolFileProvider) obj;
          return Objects.equals(getManifest(), other.getManifest())
              && Objects.equals(getSymbolFile(), other.getSymbolFile());
        }
        return false;
      }
    };
  }

  public StaticLibrary getStaticLibrary() {
    return staticLibrary;
  }

  @Override
  public String toString() {
    return String.format(
        "AndroidData(%s, %s, %s, %s, %s)", resourceDirs, assetDirs, manifest, rTxt, symbols);
  }

  @Override
  public int hashCode() {
    return Objects.hash(resourceDirs, assetDirs, manifest, rTxt, symbols);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof DependencyAndroidData)) {
      return false;
    }
    DependencyAndroidData other = (DependencyAndroidData) obj;
    return Objects.equals(other.resourceDirs, resourceDirs)
        && Objects.equals(other.assetDirs, assetDirs)
        && Objects.equals(other.rTxt, rTxt)
        && Objects.equals(other.symbols, symbols)
        && Objects.equals(other.manifest, manifest);
  }
}
