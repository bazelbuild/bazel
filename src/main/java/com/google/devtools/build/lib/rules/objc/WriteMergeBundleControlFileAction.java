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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.actions.AbstractFileWriteAction;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import com.google.devtools.build.xcode.common.TargetDeviceFamily;

import java.io.IOException;
import java.io.OutputStream;

/**
 * An action that can be used to generate a control file for the tool:
 * {@code //java/com/google/devtools/build/xcode/bundlemerge}. Note that this generates the control
 * file on-the-fly rather than eagerly. This is to prevent a copy of the bundle files and
 * .xcdatamodels from being stored for each {@code objc_binary} (or any bundle) being built.
 *
 * <p>TODO(bazel-team): Stop subclassing Action classes. Add a class to the core which lets us
 * create the control file on-the-fly and write it to a file. This may be doable by creating an
 * Action subclass in core which accepts something that generates a byte array.
 */
public class WriteMergeBundleControlFileAction extends AbstractFileWriteAction {
  private final Artifact mergedIpa;
  private final ObjcProvider objcProvider;
  private final String bundleRoot;

  /**
   * Extra files to add to the bundle besides those provided by {@link ObjcProvider#BUNDLE_FILE}.
   */
  private final Iterable<BundleMergeProtos.BundleFile> extraBundleFiles;

  private final Artifact infoplist;
  private final ObjcConfiguration objcConfiguration;
  private final Optional<Artifact> maybeActoolOutputZip;

  public WriteMergeBundleControlFileAction(RuleContext ruleContext,
      Artifact mergedIpa, ObjcProvider objcProvider,
      Iterable<BundleMergeProtos.BundleFile> extraBundleFiles,
      InfoplistMerging infoplistMerging) {
    super(ruleContext.getActionOwner(), ImmutableList.<Artifact>of(),
        /*output=*/ObjcRuleClasses.bundleMergeControlArtifact(ruleContext),
        /*makeExecutable=*/false);
    this.mergedIpa = Preconditions.checkNotNull(mergedIpa);
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
    this.bundleRoot = ObjcBinaryRule.bundleRoot(ruleContext);
    this.extraBundleFiles = Preconditions.checkNotNull(extraBundleFiles);
    this.infoplist = infoplistMerging.getPlistWithEverything();
    this.objcConfiguration = ObjcActionsBuilder.objcConfiguration(ruleContext);
    this.maybeActoolOutputZip = ObjcBinaryRule.actoolOutputZip(ruleContext, objcProvider);
  }

  public Control control() {
    BundleMergeProtos.Control.Builder control = BundleMergeProtos.Control.newBuilder()
        .addAllBundleFile(extraBundleFiles)
        .addAllBundleFile(BundleableFile.toBundleFiles(objcProvider.get(BUNDLE_FILE)))
        .addSourcePlistFile(infoplist.getExecPathString())
        // TODO(bazel-team): Add rule attributes for specifying targeted device family and minimum
        // OS version.
        .setMinimumOsVersion(MINIMUM_OS_VERSION)
        .setSdkVersion(objcConfiguration.getIosSdkVersion())
        .setPlatform(objcConfiguration.getPlatform().name())
        .setBundleRoot(bundleRoot);

    for (Artifact actoolOutputZip : maybeActoolOutputZip.asSet()) {
      control.addMergeZip(MergeZip.newBuilder()
          .setEntryNamePrefix(bundleRoot + "/")
          .setSourcePath(actoolOutputZip.getExecPathString())
          .build());
    }

    for (Xcdatamodel datamodel : objcProvider.get(XCDATAMODEL)) {
      control.addMergeZip(MergeZip.newBuilder()
          .setEntryNamePrefix(bundleRoot + "/")
          .setSourcePath(datamodel.getOutputZip().getExecPathString())
          .build());
    }
    for (TargetDeviceFamily targetDeviceFamily : TARGET_DEVICE_FAMILIES) {
      control.addTargetDeviceFamily(targetDeviceFamily.name());
    }

    control.setOutFile(mergedIpa.getExecPathString());

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
