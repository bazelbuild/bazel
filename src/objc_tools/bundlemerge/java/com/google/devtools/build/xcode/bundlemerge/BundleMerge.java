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

package com.google.devtools.build.xcode.bundlemerge;

import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.plmerge.PlistMerging.ValidationException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;

/**
 * Command-line entry point for {@link BundleMerging}. functionality. The only argument passed to
 * this command-line utility should be a control file which contains a binary serialization of the
 * {@link Control} protocol buffer.
 */
public class BundleMerge {
  private BundleMerge() {
    throw new UnsupportedOperationException("static-only");
  }

  public static void main(String[] args) throws IOException {
    if (args.length != 1) {
      System.err.println("Expect exactly one argument: path to control file");
      System.exit(1);
    }
    FileSystem fileSystem = FileSystems.getDefault();
    Control control;
    try (InputStream in = Files.newInputStream(fileSystem.getPath(args[0]))) {
      control = Control.parseFrom(in);
    }
    try {
      BundleMerging.merge(fileSystem, control);
    } catch (ValidationException e) {
      // Don't print stack traces for validation errors.
      System.err.print("\nBundle merge failed: " + e.getMessage() + "\n");
      System.exit(1);
    }
  }
}
