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

import com.android.builder.core.VariantType;
import com.android.repository.Revision;
import com.google.devtools.build.android.AaptCommandBuilder;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/** Performs linking of {@link CompiledResources} using aapt2. */
public class ResourceLinker {

  private final Path aapt2;
  private List<StaticLibrary> libraries;
  private Revision buildToolsVersion;
  private final Path workingDirectory;

  private ResourceLinker(Path aapt2, Path workingDirectory) {
    this.aapt2 = aapt2;
    this.workingDirectory = workingDirectory;
  }

  public static ResourceLinker create(Path aapt2, Path workingDirectory) {
    return new ResourceLinker(aapt2, workingDirectory);
  }

  /** Dependent static libraries to be linked to. */
  public ResourceLinker dependencies(List<StaticLibrary> libraries) {
    this.libraries = libraries;
    return this;
  }

  public ResourceLinker buildVersion(Revision buildToolsVersion) {
    this.buildToolsVersion = buildToolsVersion;
    return this;
  }

  /**
   * Statically links the {@link CompiledResources} with the dependencies to produce a {@link
   * StaticLibrary}.
   *
   * @throws IOException
   */
  public StaticLibrary linkStatically(CompiledResources resources) throws IOException {
    final Path outPath = workingDirectory.resolve("lib.ap_");
    new AaptCommandBuilder(aapt2)
        .forBuildToolsVersion(buildToolsVersion)
        .forVariantType(VariantType.LIBRARY)
        .add("link")
        .add("--manifest", resources.asManifest())
        .add("--static-lib")
        .add("--output-text-symbols", workingDirectory)
        .add("-o", outPath)
        .add("--auto-add-overlay")
        .addRepeated("-I", StaticLibrary.toPathStrings(libraries))
        .add("-R", resources.asZip())
        .execute(String.format("Linking %s", resources));
    return StaticLibrary.from(outPath);
  }
}
