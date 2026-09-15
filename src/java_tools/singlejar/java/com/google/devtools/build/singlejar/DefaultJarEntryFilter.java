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

import java.io.IOException;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.jar.JarFile;
import javax.annotation.concurrent.Immutable;

/**
 * A default filter for JAR files. It merges all services files in the {@code META-INF/services/}
 * directory. The original {@code MANIFEST} files are skipped, as are JAR signing files. Anything
 * not in the supplied path filter, an arbitrary predicate, is also skipped. To use this filter
 * properly, a new {@code MANIFEST} file should be explicitly added to the combined ZIP file.
 */
@Immutable
public class DefaultJarEntryFilter implements ZipEntryFilter {

  /** An interface to restrict which files are copied over and which are not. */
  public static interface PathFilter {
    /**
     * Returns true if an entry with the given name may be copied over.
     */
    boolean allowed(String path);
  }

  /** A filter that allows any path. */
  public static final PathFilter ANY_PATH = new PathFilter() {
    @Override
    public boolean allowed(String path) {
      return true;
    }
  };

  // ZIP timestamps have a resolution of 2 seconds, so this is the next timestamp after 1/1/1980.
  // This is only Visible for testing.
  static final Date DOS_EPOCH_PLUS_2_SECONDS =
      new GregorianCalendar(1980, 0, 1, 0, 0, 2).getTime();

  // Merge all files with a name in here:
  private static final String SERVICES_DIR = "META-INF/services/";

  // Merge all spring.handlers files.
  private static final String SPRING_HANDLERS = "META-INF/spring.handlers";

  // Merge all spring.schemas files.
  private static final String SPRING_SCHEMAS = "META-INF/spring.schemas";

  // Ignore all files with this name:
  private static final String MANIFEST_NAME = JarFile.MANIFEST_NAME;

  // Merge all protobuf extension registries.
  private static final String PROTOBUF_META = "protobuf.meta";

  // Merge all reference config files.
  private static final String REFERENCE_CONF = "reference.conf";

  protected final Date date;
  protected final Date classDate;
  protected PathFilter allowedPaths;

  public DefaultJarEntryFilter(boolean normalize, PathFilter allowedPaths) {
    this.date = normalize ? ZipCombiner.DOS_EPOCH : null;
    this.classDate = normalize ? DOS_EPOCH_PLUS_2_SECONDS : null;
    this.allowedPaths = allowedPaths;
  }

  public DefaultJarEntryFilter(boolean normalize) {
    this(normalize, ANY_PATH);
  }

  public DefaultJarEntryFilter() {
    this(true);
  }

  @Override
  public void accept(String filename, StrategyCallback callback) throws IOException {
    if (!allowedPaths.allowed(filename)) {
      callback.skip();
    } else if (filename.equals(SPRING_HANDLERS)) {
      callback.customMerge(date, new ConcatenateStrategy());
    } else if (filename.equals(SPRING_SCHEMAS)) {
      callback.customMerge(date, new ConcatenateStrategy());
    } else if (filename.equals(REFERENCE_CONF)) {
      callback.customMerge(date, new ConcatenateStrategy());
    } else if (filename.startsWith(SERVICES_DIR)) {
      // Merge all services files.
      callback.customMerge(date, new ConcatenateStrategy());
    } else if (filename.equals(MANIFEST_NAME) || filename.endsWith(".SF")
        || filename.endsWith(".DSA") || filename.endsWith(".RSA")) {
      // Ignore existing manifests and any .SF, .DSA or .RSA jar signing files.
      // TODO(bazel-team): I think we should be stricter and only skip signing
      // files from the META-INF/ directory.
      callback.skip();
    } else if (filename.endsWith(".class")) {
      // Copy .class files over, but 2 seconds ahead of the dos epoch. If it finds both source and
      // class files on the classpath, javac prefers the source file, if the class file is not newer
      // than the source file. Since we normalize the timestamps, we need to provide timestamps for
      // class files that are newer than those for the corresponding source files.
      callback.copy(classDate);
    } else if (filename.equals(PROTOBUF_META)) {
      // Merge all protobuf meta data without inserting newlines,
      // since the file is in protobuf binary format.
      callback.customMerge(date, new ConcatenateStrategy(false));
    } else {
      // Copy all other files over.
      callback.copy(date);
    }
  }
}
