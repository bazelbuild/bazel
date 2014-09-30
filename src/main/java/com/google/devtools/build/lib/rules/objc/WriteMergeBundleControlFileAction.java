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

import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.MINIMUM_OS_VERSION;
import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.TARGET_DEVICE_FAMILIES;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.NESTED_BUNDLE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.view.actions.AbstractFileWriteAction;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.VariableSubstitution;
import com.google.devtools.build.xcode.common.TargetDeviceFamily;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;

/**
 * An action that can be used to generate a control file for the tool:
 * {@code //java/com/google/devtools/build/xcode/bundlemerge}. Note that this generates the control
 * file on-the-fly rather than eagerly. This is to prevent a copy of the bundle files and
 * .xcdatamodels from being stored for each {@code objc_binary} (or any bundle) being built.
 *
 * <p>TODO(bazel-team): Stop subclassing {@link Action} classes. Add a class to the core which lets
 * us create the control file on-the-fly and write it to a file. This may be doable by creating an
 * {@link Action} subclass in core which accepts something that generates a byte array.
 */
public class WriteMergeBundleControlFileAction extends AbstractFileWriteAction {
  private final Bundling rootBundling;
  private final Artifact mergedIpa;
  private final ObjcConfiguration objcConfiguration;
  private final Map<String, String> variableSubstitutions;

  public WriteMergeBundleControlFileAction(ActionOwner actionOwner, Bundling rootBundling,
      Artifact mergedIpa, Artifact controlFile, ObjcConfiguration objcConfiguration,
      Map<String, String> variableSubstitutions) {
    super(actionOwner, /*inputs=*/ImmutableList.<Artifact>of(), controlFile,
        /*makeExecutable=*/false);
    this.rootBundling = Preconditions.checkNotNull(rootBundling);
    this.mergedIpa = Preconditions.checkNotNull(mergedIpa);
    this.objcConfiguration = Preconditions.checkNotNull(objcConfiguration);
    this.variableSubstitutions = Preconditions.checkNotNull(variableSubstitutions);
  }

  public Control control() {
    return control("Payload/", "Payload/", rootBundling);
  }

  private Control control(String mergeZipPrefix, String bundleDirPrefix, Bundling bundling) {
    ObjcProvider objcProvider = bundling.getObjcProvider();
    String bundleDir = bundleDirPrefix + bundling.getBundleDir();
    mergeZipPrefix += bundling.getBundleDir() + "/";

    BundleMergeProtos.Control.Builder control = BundleMergeProtos.Control.newBuilder()
        .addAllBundleFile(BundleableFile.toBundleFiles(bundling.getExtraBundleFiles()))
        .addAllBundleFile(BundleableFile.toBundleFiles(objcProvider.get(BUNDLE_FILE)))
        .addAllSourcePlistFile(Artifact.toExecPaths(
            bundling.getInfoplistMerging().getPlistWithEverything().asSet()))
        // TODO(bazel-team): Add rule attributes for specifying targeted device family and minimum
        // OS version.
        .setMinimumOsVersion(MINIMUM_OS_VERSION)
        .setSdkVersion(objcConfiguration.getIosSdkVersion())
        .setPlatform(objcConfiguration.getPlatform().name())
        .setBundleRoot(bundleDir);

    for (Artifact actoolzipOutput : bundling.getActoolzipOutput().asSet()) {
      control.addMergeZip(MergeZip.newBuilder()
          .setEntryNamePrefix(mergeZipPrefix)
          .setSourcePath(actoolzipOutput.getExecPathString())
          .build());
    }

    for (Xcdatamodel datamodel : objcProvider.get(XCDATAMODEL)) {
      control.addMergeZip(MergeZip.newBuilder()
          .setEntryNamePrefix(mergeZipPrefix)
          .setSourcePath(datamodel.getOutputZip().getExecPathString())
          .build());
    }
    for (TargetDeviceFamily targetDeviceFamily : TARGET_DEVICE_FAMILIES) {
      control.addTargetDeviceFamily(targetDeviceFamily.name());
    }

    // TODO(bazel-team): Should we use different variable substitutions for nested bundles?
    for (String variable : variableSubstitutions.keySet()) {
      control.addVariableSubstitution(VariableSubstitution.newBuilder()
          .setName(variable)
          .setValue(variableSubstitutions.get(variable))
          .build());
    }

    control.setOutFile(mergedIpa.getExecPathString());

    for (Artifact linkedBinary : bundling.getLinkedBinary().asSet()) {
      control.addBundleFile(BundleMergeProtos.BundleFile.newBuilder()
          .setSourceFile(linkedBinary.getExecPathString())
          .setBundlePath(bundling.getName())
          .build());
    }

    for (Bundling nestedBundling : bundling.getObjcProvider().get(NESTED_BUNDLE)) {
      control.addNestedBundle(control(mergeZipPrefix, "", nestedBundling));
    }

    return control.build();
  }

  @Override
  public void writeOutputFile(OutputStream out, EventHandler eventHandler, Executor executor)
      throws IOException, InterruptedException, ExecException {
    control().writeTo(out);
  }

  @Override
  protected String computeKey() {
    return new Fingerprint()
        .addString(control().toString())
        .hexDigest();
  }
}
