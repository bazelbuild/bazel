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

package com.google.devtools.build.lib.testutil;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Some static utility functions for testing Blaze code. In contrast to {@link TestUtils}, these
 * functions are Blaze-specific.
 */
public class BlazeTestUtils {
  private BlazeTestUtils() {}

  /**
   * Writes a FilesetRule to a String array.
   *
   * @param name the name of the rule.
   * @param out the output directory.
   * @param entries The FilesetEntry entries.
   * @return the String array of the rule.  One String for each line.
   */
  public static String[] createFilesetRule(String name, String out, String... entries) {
    return new String[] {
        String.format("Fileset(name = '%s', out = '%s',", name, out),
                      "        entries = [" +  Joiner.on(", ").join(entries) + "])"
    };
  }

  public static File undeclaredOutputDir() {
    String dir = System.getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (dir != null) {
      return new File(dir);
    }

    return TestUtils.tmpDirFile();
  }

  public static String runfilesDir() {
    String runfilesDirStr = TestUtils.getUserValue("TEST_SRCDIR");
    Preconditions.checkState(runfilesDirStr != null && runfilesDirStr.length() > 0,
        "TEST_SRCDIR unset or empty");
    return new File(runfilesDirStr).getAbsolutePath();
  }

  /** Creates an empty file, along with all its parent directories. */
  public static void makeEmptyFile(Path path) throws IOException {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    FileSystemUtils.createEmptyFile(path);
  }

  /**
   * Changes the mtime of the file "path", which must exist.  No guarantee is
   * made about the new mtime except that it is different from the previous one.
   *
   * @throws IOException if the mtime could not be read or set.
   */
  public static void changeModtime(Path path)
    throws IOException {
    long prevMtime = path.getLastModifiedTime();
    long newMtime = prevMtime;
    do {
      newMtime += 1000;
      path.setLastModifiedTime(newMtime);
    } while (path.getLastModifiedTime() == prevMtime);
  }

  public static Label convertLabel(Label label) {
    try {
      return label.getRepository().isDefault()
          ? Label.create(label.getPackageIdentifier().makeAbsolute(), label.getName())
          : label;
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Creates a list of arguments to pass to Bazel, with flags necessary for Bazel to work properly
   * appended to the original {@code args} array.
   */
  public static ArrayList<String> makeArgs(String... args) {
    ArrayList<String> result =
        new ArrayList<>(args.length + TestConstants.PRODUCT_SPECIFIC_FLAGS.size());
    Collections.addAll(result, args);
    result.addAll(TestConstants.PRODUCT_SPECIFIC_FLAGS);
    return result;
  }
}
