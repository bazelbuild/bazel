// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8;

import com.google.common.base.Ascii;
import java.nio.file.Path;

/** File related utilities */
public class FileUtils {
  public static final String AAR_EXTENSION = ".aar";
  public static final String APK_EXTENSION = ".apk";
  public static final String CLASS_EXTENSION = ".class";
  public static final String DEX_EXTENSION = ".dex";
  public static final String JAR_EXTENSION = ".jar";
  public static final String ZIP_EXTENSION = ".zip";
  public static final String MODULE_INFO_CLASS = "module-info.class";
  public static final String META_INF = "meta-inf";

  private static boolean hasExtension(String name, String extension) {
    return Ascii.toLowerCase(name).endsWith(extension);
  }

  private static boolean hasExtension(Path path, String extension) {
    return hasExtension(path.getFileName().toString(), extension);
  }

  public static boolean isDexFile(Path path) {
    return hasExtension(path, DEX_EXTENSION);
  }

  public static boolean isClassFile(String name) {
    name = Ascii.toLowerCase(name);
    // Android does not support Java 9 module, thus skip module-info.
    if (name.equals(MODULE_INFO_CLASS)) {
      return false;
    }
    if (name.startsWith(META_INF) || name.startsWith("/" + META_INF)) {
      return false;
    }
    return name.endsWith(CLASS_EXTENSION);
  }

  public static boolean isClassFile(Path path) {
    return isClassFile(path.getFileName().toString());
  }

  public static boolean isJarFile(String name) {
    return hasExtension(name, JAR_EXTENSION);
  }

  public static boolean isJarFile(Path path) {
    return hasExtension(path, JAR_EXTENSION);
  }

  public static boolean isZipFile(String name) {
    return hasExtension(name, ZIP_EXTENSION);
  }

  public static boolean isZipFile(Path path) {
    return hasExtension(path, ZIP_EXTENSION);
  }

  public static boolean isApkFile(String name) {
    return hasExtension(name, APK_EXTENSION);
  }

  public static boolean isApkFile(Path path) {
    return hasExtension(path, APK_EXTENSION);
  }

  public static boolean isAarFile(String name) {
    return hasExtension(name, AAR_EXTENSION);
  }

  public static boolean isAarFile(Path path) {
    return hasExtension(path, AAR_EXTENSION);
  }

  public static boolean isArchive(Path path) {
    return isApkFile(path) || isJarFile(path) || isZipFile(path) || isAarFile(path);
  }

  private FileUtils() {}
}
