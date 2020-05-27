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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import java.nio.file.Path;
import java.util.zip.ZipEntry;

/** A indexer for the entry items in the Jar files in the boot class path. */
@AutoValue
public abstract class BootClassPathDigest {

  private static final Splitter PACKAGE_BINARY_DELIMITER = Splitter.on('/').trimResults();

  abstract ImmutableList<Path> bootClassPaths();

  public static BootClassPathDigest create(ImmutableList<Path> bootClassPaths) {
    return new AutoValue_BootClassPathDigest(bootClassPaths);
  }

  @Memoized
  public ImmutableSet<String> allResourceEntryNames() {
    return bootClassPaths().stream()
        .flatMap(path -> JarItem.newJarFile(path.toFile()).stream())
        .map(ZipEntry::getName)
        .collect(toImmutableSet());
  }

  public final int resourceEntrySize() {
    return allResourceEntryNames().size();
  }

  public final ImmutableList<String> listPackageLeadingPrefixes() {
    return allResourceEntryNames().stream()
        .map(PACKAGE_BINARY_DELIMITER::splitToList)
        .map(segs -> Iterables.getFirst(segs, "<empty>"))
        .distinct()
        .sorted()
        .collect(toImmutableList());
  }

  public final boolean containsResourceEntry(String resourceEntryName) {
    return allResourceEntryNames().contains(resourceEntryName);
  }

  public final boolean containsType(ClassName className) {
    return containsResourceEntry(className.classFilePathName());
  }
}
