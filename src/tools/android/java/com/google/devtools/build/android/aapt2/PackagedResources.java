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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.android.ResourcesZip;
import java.nio.file.Path;

/** Represents the packaged, flattened resources. */
@AutoValue
public abstract class PackagedResources {

  public abstract Path apk();

  public abstract Path proto();

  public abstract Path rTxt();

  public abstract Path proguardConfig();

  public abstract Path mainDexProguard();

  public abstract Path javaSourceDirectory();

  abstract Path resourceIds();

  public abstract Path attributes();

  public abstract Path packages();

  public static PackagedResources of(
      Path outPath,
      Path protoPath,
      Path rTxt,
      Path proguardConfig,
      Path mainDexProguard,
      Path javaSourceDirectory,
      Path resourceIds,
      Path attributes,
      Path packages) {
    return new AutoValue_PackagedResources(
        outPath,
        protoPath,
        rTxt,
        proguardConfig,
        mainDexProguard,
        javaSourceDirectory,
        resourceIds,
        attributes,
        packages);
  }

  public ResourcesZip asArchive() {
    return ResourcesZip.fromApkWithProto(proto(), attributes(), resourceIds(), packages());
  }
}
