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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.VariableSubstitution;

import java.io.InputStream;
import java.util.Map;

/**
 * A byte source that can be used to generate a control file for the tool:
 * {@code //java/com/google/devtools/build/xcode/bundlemerge}. Note that this generates the control
 * proto and bytes on-the-fly rather than eagerly. This is to prevent a copy of the bundle files and
 * .xcdatamodels from being stored for each {@code objc_binary} (or any bundle) being built.
 */
// TODO(bazel-team): Move the logic in this class to Bundling (as a .toControl method).
final class BundleMergeControlBytes extends ByteSource {
  private final Bundling rootBundling;
  private final Artifact mergedIpa;
  private final ObjcConfiguration objcConfiguration;
  private final ImmutableSet<TargetDeviceFamily> families;

  public BundleMergeControlBytes(
      Bundling rootBundling, Artifact mergedIpa, ObjcConfiguration objcConfiguration,
      ImmutableSet<TargetDeviceFamily> families) {
    this.rootBundling = Preconditions.checkNotNull(rootBundling);
    this.mergedIpa = Preconditions.checkNotNull(mergedIpa);
    this.objcConfiguration = Preconditions.checkNotNull(objcConfiguration);
    this.families = Preconditions.checkNotNull(families);
  }

  @Override
  public InputStream openStream() {
    return control("", rootBundling)
        .toByteString()
        .newInput();
  }

  private Control control(String mergeZipPrefix, Bundling bundling) {
    mergeZipPrefix += bundling.getBundleDir() + "/";

    BundleMergeProtos.Control.Builder control = BundleMergeProtos.Control.newBuilder()
        .addAllBundleFile(BundleableFile.toBundleFiles(bundling.getBundleFiles()))
        // TODO(bazel-team): This should really be bundling.getBundleInfoplistInputs since (most of)
        // those are editable, whereas this is usually the programatically merged plist. If we pass
        // the sources here though, any synthetic data (generated plists with blaze-derived values)
        // should be passed as well.
        .addAllSourcePlistFile(Artifact.toExecPaths(bundling.getBundleInfoplist().asSet()))
        // TODO(bazel-team): Add rule attribute for specifying targeted device family
        .setMinimumOsVersion(bundling.getMinimumOsVersion())
        .setSdkVersion(objcConfiguration.getIosSdkVersion())
        .setPlatform(objcConfiguration.getBundlingPlatform().name())
        .setBundleRoot(bundling.getBundleDir());

    for (Artifact mergeZip : bundling.getMergeZips()) {
      control.addMergeZip(MergeZip.newBuilder()
          .setEntryNamePrefix(mergeZipPrefix)
          .setSourcePath(mergeZip.getExecPathString())
          .build());
    }

    for (TargetDeviceFamily targetDeviceFamily : families) {
      control.addTargetDeviceFamily(targetDeviceFamily.name());
    }

    Map<String, String> variableSubstitutions = bundling.variableSubstitutions();
    for (String variable : variableSubstitutions.keySet()) {
      control.addVariableSubstitution(VariableSubstitution.newBuilder()
          .setName(variable)
          .setValue(variableSubstitutions.get(variable))
          .build());
    }

    control.setOutFile(mergedIpa.getExecPathString());

    for (Artifact linkedBinary : bundling.getCombinedArchitectureBinary().asSet()) {
      control
          .addBundleFile(BundleMergeProtos.BundleFile.newBuilder()
              .setSourceFile(linkedBinary.getExecPathString())
              .setBundlePath(bundling.getName())
              .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
              .build())
          .setExecutableName(bundling.getName());
    }

    for (Bundling nestedBundling : bundling.getNestedBundlings()) {
      if (nestedBundling.getArchitecture().equals(bundling.getArchitecture())) {
        control.addNestedBundle(control(mergeZipPrefix, nestedBundling));
      }
    }
    
    if (bundling.getPrimaryBundleId()  != null) {
      control.setPrimaryBundleIdentifier(bundling.getPrimaryBundleId());
    }
    
    if (bundling.getFallbackBundleId() != null) {
      control.setFallbackBundleIdentifier(bundling.getFallbackBundleId());
    }
    
    return control.build();
  }
}
