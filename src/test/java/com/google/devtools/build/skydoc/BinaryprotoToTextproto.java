// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.TextFormat;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

/**
 * Converts Stardoc binaryproto files (read from stdin) to textproto format (printed to stdout), for
 * more convenient use in tests.
 */
final class BinaryprotoToTextproto {

  public static void main(String[] args) throws IOException {
    binaryprotoToTextproto(System.in, System.out);
  }

  private static void binaryprotoToTextproto(
      InputStream binaryprotoInput, OutputStream textprotoOutput) throws IOException {
    ModuleInfo moduleInfo =
        ModuleInfo.parseFrom(binaryprotoInput, ExtensionRegistry.getEmptyRegistry());
    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(textprotoOutput, UTF_8));
    TextFormat.printer().print(moduleInfo, writer);
    writer.flush();
  }

  private BinaryprotoToTextproto() {}
}
