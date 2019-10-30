// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.cmdline;

import com.google.devtools.build.lib.vfs.PathFragment;

/** Constants associated with {@code Label}s */
public class LabelConstants {
  public static final PathFragment EXTERNAL_PACKAGE_NAME = PathFragment.create("external");
  public static final PackageIdentifier EXTERNAL_PACKAGE_IDENTIFIER =
      PackageIdentifier.createInMainRepo(EXTERNAL_PACKAGE_NAME);
  public static final PathFragment EXTERNAL_PATH_PREFIX = PathFragment.create("external");
  public static final PathFragment WORKSPACE_FILE_NAME = PathFragment.create("WORKSPACE");
  public static final String DEFAULT_REPOSITORY_DIRECTORY = "__main__";

  /**
   * Whether Bazel should check (true) or trust (false) the casing of Labels.
   *
   * <p>Package "//foo" exists if "foo/BUILD" exists, i.e. stat("foo/BUILD") succeeds.
   *
   * <p>If "foo/BUILD" exists on a case-sensitive filesystem (e.g. Ext4), stat("foo/BUILD") succeeds
   * while stat("Foo/BUILD") and stat("FOO/build") fail. But on a case-ignoring filesystem (e.g.
   * APFS and NTFS), all three calls succeed, so "//Foo" and "//FOO" also appear to exist (falsely).
   */
  public static final boolean CHECK_CASING = true;  // TODO(laszlocsomor): Undo this.
      //"1".equals(System.getProperty("bazel.check_label_casing"));
}
