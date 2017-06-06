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

import com.google.common.io.Files;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.util.TreeSet;

/** A command that prints an embedded license text. */
@Command(
  name = "license",
  allowResidue = true,
  mustRunInWorkspace = false,
  shortDescription = "Prints the license of this software.",
  help = "Prints the license of this software.\n\n%{options}"
)
public class LicenseCommand implements BlazeCommand {

  private static final TreeSet<String> JAVA_LICENSE_FILES =
      new TreeSet<>(String.CASE_INSENSITIVE_ORDER);

  static {
    JAVA_LICENSE_FILES.add("ASSEMBLY_EXCEPTION");
    JAVA_LICENSE_FILES.add("DISCLAIMER");
    JAVA_LICENSE_FILES.add("LICENSE");
    JAVA_LICENSE_FILES.add("THIRD_PARTY_README");
  }

  public static boolean isSupported() {
    return ResourceFileLoader.resourceExists(LicenseCommand.class, "LICENSE");
  }

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
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
        env.getDirectories().getEmbeddedBinariesRoot().getRelative("embedded_tools/jdk");
    if (bundledJdk.exists()) {
      outErr.printOutLn(
          "This binary comes with a bundled JDK, which contains the following license files:\n");
      printJavaLicenseFiles(outErr, bundledJdk);
    }

    Path bundledJre =
        env.getDirectories().getEmbeddedBinariesRoot().getRelative("embedded_tools/jre");
    if (bundledJre.exists()) {
      outErr.printOutLn(
          "This binary comes with a bundled JRE, which contains the following license files:\n");
      printJavaLicenseFiles(outErr, bundledJre);
    }

    return ExitCode.SUCCESS;
  }

  private static void printJavaLicenseFiles(OutErr outErr, Path bundledJdkOrJre) {
    try {
      for (Path path : bundledJdkOrJre.getDirectoryEntries()) {
        if (JAVA_LICENSE_FILES.contains(path.getBaseName().toLowerCase())) {
          outErr.printOutLn(path.getPathString() + ":\n");
          Files.copy(path.getPathFile(), outErr.getOutputStream());
          outErr.printOutLn("\n");
        }
      }
    } catch (IOException e) {
      throw new IllegalStateException(
          "I/O error while trying to print license file of bundled JDK or JRE: " + e.getMessage(),
          e);
    }
  }

  @Override
  public void editOptions(OptionsParser optionsParser) {}
}
