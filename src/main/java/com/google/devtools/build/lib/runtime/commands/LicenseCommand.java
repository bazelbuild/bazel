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
package com.google.devtools.build.lib.runtime.commands;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;

/** A command that prints an embedded license text. */
@Command(
  name = "license",
  allowResidue = true,
  mustRunInWorkspace = false,
  shortDescription = "Prints the license of this software.",
  help = "Prints the license of this software.\n\n%{options}"
)
public class LicenseCommand implements BlazeCommand {

  private static final ImmutableSet<String> JAVA_LICENSE_FILES =
      ImmutableSet.of("ASSEMBLY_EXCEPTION", "DISCLAIMER", "LICENSE", "THIRD_PARTY_README");

  public static boolean isSupported() {
    return ResourceFileLoader.resourceExists(LicenseCommand.class, "LICENSE");
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    env.getEventBus().post(new NoBuildEvent());
    OutErr outErr = env.getReporter().getOutErr();

    outErr.printOutLn("Licenses of all components included in this binary:\n");

    try {
      outErr.printOutLn(ResourceFileLoader.loadResource(this.getClass(), "LICENSE"));
    } catch (IOException e) {
      throw new IllegalStateException(
          "I/O error while trying to print 'LICENSE' resource: " + e.getMessage(), e);
    }

    Path bundledJdk =
        env.getDirectories()
            .getEmbeddedBinariesRoot()
            .getRelative("embedded_tools/jdk")
            .getPathFile()
            .toPath();
    if (Files.exists(bundledJdk)) {
      outErr.printOutLn(
          "This binary comes with a bundled JDK, which contains the following license files:\n");
      printJavaLicenseFiles(outErr, bundledJdk);
    }

    Path bundledJre =
        env.getDirectories()
            .getEmbeddedBinariesRoot()
            .getRelative("embedded_tools/jre")
            .getPathFile()
            .toPath();
    if (Files.exists(bundledJre)) {
      outErr.printOutLn(
          "This binary comes with a bundled JRE, which contains the following license files:\n");
      printJavaLicenseFiles(outErr, bundledJre);
    }

    return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
  }

  private static void printJavaLicenseFiles(OutErr outErr, Path bundledJdkOrJre) {
    try {
      Files.walkFileTree(
          bundledJdkOrJre,
          new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path path, BasicFileAttributes basicFileAttributes)
                throws IOException {
              if (JAVA_LICENSE_FILES.contains(path.getFileName().toString())) {
                outErr.printOutLn(path + ":\n");
                Files.copy(path, outErr.getOutputStream());
                outErr.printOutLn("\n");
              }
              return super.visitFile(path, basicFileAttributes);
            }
          });
    } catch (IOException e) {
      throw new UncheckedIOException(
          "I/O error while trying to print license file of bundled JDK or JRE: " + e.getMessage(),
          e);
    }
  }
}
