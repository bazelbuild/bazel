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

package com.google.devtools.build.singlejar;

import com.google.devtools.build.zip.ExtraData;

import java.io.IOException;
import java.util.Date;

/**
 * Provides utilities for using ZipCombiner to pack up Jar files.
 */
public final class JarUtils {
  private static final String MANIFEST_DIRECTORY = "META-INF/";
  private static final short MAGIC_JAR_ID = (short) 0xCAFE;
  private static final ExtraData[] MAGIC_JAR_ID_EXTRA_ENTRIES =
      new ExtraData[] { new ExtraData(MAGIC_JAR_ID, new byte[0]) };

  /**
   * Adds META-INF directory through ZipCombiner with the given date and the
   * magic jar ID.
   *
   * @throws IOException if {@link ZipCombiner#addDirectory(String, Date, ExtraData[])}
   *                     throws an IOException.
   */
  public static void addMetaInf(ZipCombiner combiner, Date date) throws IOException {
    combiner.addDirectory(MANIFEST_DIRECTORY, date, MAGIC_JAR_ID_EXTRA_ENTRIES);
  }
}
