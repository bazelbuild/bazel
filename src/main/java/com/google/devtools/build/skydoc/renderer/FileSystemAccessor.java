// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skydoc.renderer;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Implementation of {@link ProtoFileAccessor} which uses the real filesystem. */
public class FileSystemAccessor implements ProtoFileAccessor {

  @Override
  public byte[] getProtoContent(String inputPathString) throws IOException {
    Path inputPath = Paths.get(inputPathString);
    byte[] inputContent = Files.readAllBytes(inputPath);
    return inputContent;
  }

  @Override
  public boolean fileExists(String pathString) {
    return Files.exists(Paths.get(pathString));
  }

  @Override
  public void writeToOutputLocation(String outputPathString, byte[] content) throws IOException {
    try (FileOutputStream outputStream = new FileOutputStream(outputPathString)) {
      for (byte byteContent : content) {
        outputStream.write(byteContent);
      }
    }
  }
}
