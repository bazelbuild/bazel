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
package com.google.devtools.build.android.aapt2;

import com.google.devtools.build.android.AndroidResourceOutputs;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Represents the packaged, flattened resources. */
public class PackagedResources {

  private final Path apk;
  private final Path rTxt;
  private final Path proguardConfig;
  private final Path mainDexProguard;
  private final Path javaSourceDirectory;

  private PackagedResources(
      Path apk, Path rTxt, Path proguardConfig, Path mainDexProguard, Path javaSourceDirectory) {
    this.apk = apk;
    this.rTxt = rTxt;
    this.proguardConfig = proguardConfig;
    this.mainDexProguard = mainDexProguard;
    this.javaSourceDirectory = javaSourceDirectory;
  }

  public PackagedResources copyPackageTo(Path packagePath) throws IOException {
    return new PackagedResources(
        copy(apk, packagePath), rTxt, proguardConfig, mainDexProguard, javaSourceDirectory);
  }

  public PackagedResources copyRTxtTo(Path rOutput) throws IOException {
    if (rOutput == null) {
      return this;
    }
    return new PackagedResources(
        apk, copy(rTxt, rOutput), proguardConfig, mainDexProguard, javaSourceDirectory);
  }

  private Path copy(Path from, Path out) throws IOException {
    Files.createDirectories(out.getParent());
    Files.copy(from, out);
    return out;
  }

  public PackagedResources copyProguardTo(Path proguardOut) throws IOException {
    if (proguardOut == null) {
      return this;
    }
    return new PackagedResources(
        apk, rTxt, copy(proguardConfig, proguardOut), mainDexProguard, javaSourceDirectory);
  }

  public PackagedResources copyMainDexProguardTo(Path mainDexProguardOut) throws IOException {
    if (mainDexProguardOut == null) {
      return this;
    }
    return of(
        apk, rTxt, proguardConfig, copy(mainDexProguard, mainDexProguardOut), javaSourceDirectory);
  }

  public PackagedResources createSourceJar(Path sourceJarPath) throws IOException {
    if (sourceJarPath == null) {
      return this;
    }
    AndroidResourceOutputs.createSrcJar(javaSourceDirectory, sourceJarPath, false);
    return of(apk, rTxt, proguardConfig, mainDexProguard, sourceJarPath);
  }

  public static PackagedResources of(
      Path outPath, Path rTxt, Path proguardConfig, Path mainDexProguard, Path javaSourceDirectory)
      throws IOException {
    return new PackagedResources(
        outPath, rTxt, proguardConfig, mainDexProguard, javaSourceDirectory);
  }
}
