// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;

import java.io.InputStream;

/**
 * A byte source that can be used to generate a control file for the tool bundlemerge . 
 * Note that this generates the control proto and bytes on-the-fly rather than eagerly. 
 * This is to prevent a copy of the bundle files and .xcdatamodels from being stored for 
 * each {@code objc_binary} (or any bundle) being built.
 */
// TODO(bazel-team): Move the logic in this class to Bundling (as a .toControl method).
final class BundleMergeControlBytes extends ByteSource {
  private final Bundling rootBundling;
  private final Artifact mergedIpa;
  private final AppleConfiguration appleConfiguration;

  public BundleMergeControlBytes(
      Bundling rootBundling, Artifact mergedIpa, AppleConfiguration appleConfiguration) {
    this.rootBundling = Preconditions.checkNotNull(rootBundling);
    this.mergedIpa = Preconditions.checkNotNull(mergedIpa);
    this.appleConfiguration = Preconditions.checkNotNull(appleConfiguration);
  }

  @Override
  public InputStream openStream() {
    return control("", rootBundling).toByteString().newInput();
  }

  private Control control(String mergeZipPrefix, Bundling bundling) {
    mergeZipPrefix += bundling.getBundleDir() + "/";

    BundleMergeProtos.Control.Builder control =
        BundleMergeProtos.Control.newBuilder()
            .addAllBundleFile(BundleableFile.toBundleFiles(bundling.getBundleFiles()))
            // TODO(bazel-team): Add rule attribute for specifying targeted device family
            .setMinimumOsVersion(bundling.getMinimumOsVersion().toString())
            .setSdkVersion(appleConfiguration.getIosSdkVersion().toString())
            .setPlatform(appleConfiguration.getBundlingPlatform().name())
            .setBundleRoot(bundling.getBundleDir());

    if (bundling.getBundleInfoplist().isPresent()) {
      control.setBundleInfoPlistFile((bundling.getBundleInfoplist().get().getExecPathString()));
    }
    
    for (Artifact mergeZip : bundling.getMergeZips()) {
      control.addMergeZip(MergeZip.newBuilder()
          .setEntryNamePrefix(mergeZipPrefix)
          .setSourcePath(mergeZip.getExecPathString())
          .build());
    }

    control.setOutFile(mergedIpa.getExecPathString());

    for (Artifact linkedBinary : bundling.getCombinedArchitectureBinary().asSet()) {
      control.addBundleFile(
          BundleMergeProtos.BundleFile.newBuilder()
              .setSourceFile(linkedBinary.getExecPathString())
              .setBundlePath(bundling.getName())
              .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
              .build());
    }

    for (Bundling nestedBundling : bundling.getNestedBundlings()) {
      if (nestedBundling.getArchitecture().equals(bundling.getArchitecture())) {
        control.addNestedBundle(control(mergeZipPrefix, nestedBundling));
      }
    }
    
    if (bundling.getPrimaryBundleId() != null) {
      control.setPrimaryBundleIdentifier(bundling.getPrimaryBundleId());
    }
    
    if (bundling.getFallbackBundleId() != null) {
      control.setFallbackBundleIdentifier(bundling.getFallbackBundleId());
    }
    
    return control.build();
  }
}
