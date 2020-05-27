/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.google.devtools.build.android.desugar.io;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.collect.ImmutableSet;
import java.nio.file.Path;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/** A jar file wrapper with jar entry analysis utilities. */
@AutoValue
public abstract class JarDigest {

  abstract JarFile jarFile();

  public static JarDigest create(JarFile jarFile) {
    return new AutoValue_JarDigest(jarFile);
  }

  public static JarDigest fromPath(Path jarPath) {
    return create(JarItem.newJarFile(jarPath.toFile()));
  }

  public final boolean hasPackagePrefix(String prefix) {
    return getAllClassFilePackagePrefixes().contains(prefix);
  }

  @Memoized
  ImmutableSet<String> getAllClassFilePackagePrefixes() {
    return jarFile().stream()
        .map(JarEntry::getName)
        .filter(name -> name.endsWith(".class"))
        .flatMap(name -> getAllPackagePrefixes(name).stream())
        .collect(toImmutableSet());
  }

  /**
   * Returns true if the given jar in class path is a platform jar, which will be added to the boot
   * class path.
   */
  public boolean isPlatformJar() {
    // Configured per b/153106333
    return hasPackagePrefix("android/car/content/") && !hasPackagePrefix("android/car/test/");
  }

  private static ImmutableSet<String> getAllPackagePrefixes(String resourceName) {
    ImmutableSet.Builder<String> prefixes = ImmutableSet.builder();
    for (int i = 0; i < resourceName.length(); i++) {
      if (resourceName.charAt(i) == '/') {
        prefixes.add(resourceName.substring(0, i + 1));
      }
    }
    return prefixes.build();
  }
}
