/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.io;

import com.google.auto.value.AutoValue;
import com.google.common.collect.Streams;
import java.io.File;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Stream;

/** A bundle of jar entry with its affiliated jar file. */
@AutoValue
public abstract class JarItem {

  public abstract JarFile jarFile();

  public abstract JarEntry jarEntry();

  public static JarItem create(JarFile jarFile, JarEntry jarEntry) {
    return new AutoValue_JarItem(jarFile, jarEntry);
  }

  public static Stream<JarItem> jarItemStream(JarFile jarFile) {
    return Streams.zip(Stream.generate(() -> jarFile), jarFile.stream(), JarItem::create);
  }

  public static Stream<JarItem> jarItemStream(Path jarFilePath) {
    return jarItemStream(newJarFile(jarFilePath.toFile()));
  }

  public static JarFile newJarFile(File file) {
    try {
      return new JarFile(file);
    } catch (IOException e) {
      throw new IOError(e);
    }
  }

  public final Path jarPath() {
    return Paths.get(jarFile().getName());
  }

  public final String jarEntryName() {
    return jarEntry().getName();
  }

  public final InputStream getInputStream() {
    try {
      return jarFile().getInputStream(jarEntry());
    } catch (IOException e) {
      throw new IOError(e);
    }
  }
}
