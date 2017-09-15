// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Utility class to create Windows native launcher */
public final class NativeLauncherUtil {

  private NativeLauncherUtil() {}

  /** Write a string to launch info buffer. */
  public static void writeLaunchInfo(ByteArrayOutputStream launchInfo, String value)
      throws IOException {
    launchInfo.write(value.getBytes(UTF_8));
  }

  /** Write a key-value pair launch info to buffer. */
  public static void writeLaunchInfo(ByteArrayOutputStream launchInfo, String key, String value)
      throws IOException {
    launchInfo.write(key.getBytes(UTF_8));
    launchInfo.write('=');
    launchInfo.write(value.getBytes(UTF_8));
    launchInfo.write('\0');
  }

  /**
   * Write a key-value pair launch info to buffer. The method construct the value from a list of
   * String separated by delimiter.
   */
  public static void writeLaunchInfo(
      ByteArrayOutputStream launchInfo,
      String key,
      final Iterable<String> valueList,
      char delimiter)
      throws IOException {
    launchInfo.write(key.getBytes(UTF_8));
    launchInfo.write('=');
    boolean isFirst = true;
    for (String value : valueList) {
      if (!isFirst) {
        launchInfo.write(delimiter);
      } else {
        isFirst = false;
      }
      launchInfo.write(value.getBytes(UTF_8));
    }
    launchInfo.write('\0');
  }

  /**
   * Write the size of all the launch info as a 64-bit integer at the end of the output stream in
   * little endian.
   */
  public static void writeDataSize(ByteArrayOutputStream launchInfo) throws IOException {
    long dataSize = launchInfo.size();
    ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES);
    // All Windows versions are little endian.
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    buffer.putLong(dataSize);
    launchInfo.write(buffer.array());
  }

  /**
   * The launcher file consists of a base launcher binary and the launch information appended to the
   * binary.
   *
   * @param ruleContext The rule context.
   * @param launcher The exe launcher we are going to build.
   * @param launchInfo The launch info to be appended.
   */
  public static void createNativeLauncherActions(
      RuleContext ruleContext, Artifact launcher, ByteArrayOutputStream launchInfo) {
    createNativeLauncherActions(ruleContext, launcher, ByteSource.wrap(launchInfo.toByteArray()));
  }

  public static void createNativeLauncherActions(
      RuleContext ruleContext, Artifact launcher, ByteSource launchInfo) {
    Artifact launchInfoFile =
        ruleContext.getRelatedArtifact(launcher.getRootRelativePath(), ".launch_info");

    ruleContext.registerAction(
        new BinaryFileWriteAction(
            ruleContext.getActionOwner(), launchInfoFile, launchInfo, /*makeExecutable=*/ false));

    Artifact baseLauncherBinary = ruleContext.getPrerequisiteArtifact("$launcher", Mode.HOST);

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(baseLauncherBinary)
            .addInput(launchInfoFile)
            .addOutput(launcher)
            .setShellCommand(
                "cmd.exe /c \"copy /Y /B "
                    + baseLauncherBinary.getExecPathString().replace('/', '\\')
                    + "+"
                    + launchInfoFile.getExecPathString().replace('/', '\\')
                    + " "
                    + launcher.getExecPathString().replace('/', '\\')
                    + " > nul\"")
            .useDefaultShellEnvironment()
            .setMnemonic("BuildNativeLauncher")
            .build(ruleContext));
  }
}
