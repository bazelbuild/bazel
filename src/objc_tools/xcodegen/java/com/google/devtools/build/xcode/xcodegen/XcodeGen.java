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

package com.google.devtools.build.xcode.xcodegen;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.Control;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import com.facebook.buck.apple.xcode.xcodeproj.PBXProject;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;

/**
 * Entry-point for the command-line Xcode project generator.
 */
public class XcodeGen {
  /**
   * Options for {@link XcodeGen}.
   */
  public static class XcodeGenOptions extends OptionsBase {
    @Option(
        name = "control",
        help = "Path to a control file, which contains only a binary serialized instance of "
            + "the Control protocol buffer. Required.",
        defaultValue = "null")
    public String control;
  }

  public static void main(String[] args) throws IOException, OptionsParsingException {
    OptionsParser parser = OptionsParser.newOptionsParser(XcodeGenOptions.class);
    parser.parse(args);
    XcodeGenOptions options = parser.getOptions(XcodeGenOptions.class);
    if (options.control == null) {
      throw new IllegalArgumentException("--control must be specified\n"
          + Options.getUsage(XcodeGenOptions.class));
    }
    FileSystem fileSystem = FileSystems.getDefault();

    Control controlPb;
    try (InputStream in = Files.newInputStream(fileSystem.getPath(options.control))) {
      controlPb = Control.parseFrom(in);
    }
    Path pbxprojPath = fileSystem.getPath(controlPb.getPbxproj());

    Iterator<String> srcList = allSourceFilePaths(controlPb).iterator();
    Path workspaceRoot;
    if (!srcList.hasNext()) {
      workspaceRoot = XcodeprojGeneration.relativeWorkspaceRoot(pbxprojPath);
    } else {
      // Get the absolute path to the workspace root.

      // TODO(bazel-team): Remove this hack, possibly by converting Xcodegen to be run with
      // "bazel run" and using RUNFILES to get the workspace root. For now, this is needed to work
      // around Xcode's handling of symlinks not playing nicely with how Bazel stores output
      // artifacts in /private/var/tmp. This means a relative path from .xcodeproj in bazel-out to
      // the workspace root in .xcodeproj will not work properly at certain times during
      // Xcode/xcodebuild execution. Walking up the path of a known source file prevents having
      // to reason about a file that might only be accessible through a symlink, like a tools jar.
      Path relSourceFilePath = fileSystem.getPath(srcList.next());
      Path absSourceFilePath = relSourceFilePath.toAbsolutePath();
      workspaceRoot = absSourceFilePath;
      for (int i = 0; i < relSourceFilePath.getNameCount(); i++) {
        workspaceRoot = workspaceRoot.getParent();
      }
    }

    try (OutputStream out = Files.newOutputStream(pbxprojPath)) {
      // This workspace root here is relative to the PWD, so that the .xccurrentversion
      // files can actually be read. The other workspaceRoot is relative to the .xcodeproj
      // root or is absolute.
      Path relativeWorkspaceRoot = fileSystem.getPath(".");
      PBXProject project = XcodeprojGeneration.xcodeproj(
          workspaceRoot, controlPb,
          ImmutableList.of(
              new CurrentVersionSetter(relativeWorkspaceRoot),
              new PbxReferencesGrouper(fileSystem)));
      XcodeprojGeneration.write(out, project);
    }
  }

  private static Iterable<String> allSourceFilePaths(Control control) {
    return Iterables.concat(
        Iterables.transform(control.getTargetList(),
                            new Function<TargetControl, List<String>>() {
                              public List<String> apply(TargetControl tc) {
                                return tc.getSourceFileList();
                              }
                            }));
  }
}
