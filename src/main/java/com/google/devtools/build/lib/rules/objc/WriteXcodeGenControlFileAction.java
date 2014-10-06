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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.view.actions.AbstractFileWriteAction;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.Control;

import java.io.IOException;
import java.io.OutputStream;

/**
 * An action that can be used to generate a control file for the tool:
 * {@link com.google.devtools.build.xcode.xcodegen.XcodeGen}.
 */
public class WriteXcodeGenControlFileAction extends AbstractFileWriteAction {
  private final XcodeProvider xcodeProvider;
  private final Artifact pbxproj;

  public WriteXcodeGenControlFileAction(ActionOwner owner, Artifact output,
      XcodeProvider xcodeProvider, Artifact pbxproj) {
    super(owner, ImmutableList.<Artifact>of(), output, /*makeExecutable=*/false);
    this.xcodeProvider = Preconditions.checkNotNull(xcodeProvider);
    this.pbxproj = Preconditions.checkNotNull(pbxproj);
  }

  public Control control() {
    return XcodeGenProtos.Control.newBuilder()
        .setPbxproj(pbxproj.getExecPathString())
        .addAllTarget(xcodeProvider.getTargets())
        .build();
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
