// Copyright 2024 The Bazel Authors. All rights reserved.
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
package diskcache;

import static java.lang.Math.min;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.remote.disk.DiskCacheGarbageCollector;
import com.google.devtools.build.lib.remote.disk.DiskCacheGarbageCollector.CollectionPolicy;
import com.google.devtools.build.lib.remote.disk.DiskCacheGarbageCollector.CollectionStats;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.windows.WindowsFileSystem;
import com.google.devtools.common.options.Converters.ByteSizeConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.time.Duration;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Standalone disk cache garbage collection utility. */
public final class Gc {

  private Gc() {}

  /** Command line options. */
  public static final class Options extends OptionsBase {

    @Option(
        name = "disk_cache",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Path to disk cache.")
    public String diskCache;

    @Option(
        name = "max_size",
        defaultValue = "0",
        converter = ByteSizeConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "The target size for the disk cache. If set to a positive value, older entries will be"
                + " deleted as required to reach this size.")
    public long maxSize;

    @Option(
        name = "max_age",
        defaultValue = "0",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "The target age for the disk cache. If set to a positive value, entries exceeding this"
                + " age will be deleted.")
    public Duration maxAge;
  }

  private static final ExecutorService executorService =
      Executors.newFixedThreadPool(
          min(4, Runtime.getRuntime().availableProcessors()),
          new ThreadFactoryBuilder().setNameFormat("disk-cache-gc-%d").build());

  public static void main(String[] args) throws Exception {
    OptionsParser op = OptionsParser.builder().optionsClasses(Options.class).build();
    op.parseAndExitUponError(args);

    Options options = op.getOptions(Options.class);

    if (options.diskCache == null) {
      System.err.println("--disk_cache must be specified.");
      System.exit(1);
    }

    if (options.maxSize <= 0 && options.maxAge.isZero()) {
      System.err.println(
          "At least one of --max_size or --max_age must be set to a positive value.");
      System.exit(1);
    }

    var root = getFileSystem().getPath(options.diskCache);
    if (!root.isDirectory()) {
      System.err.println("Expected --disk_cache to exist and be a directory.");
      System.exit(1);
    }

    var policy =
        new CollectionPolicy(
            options.maxSize == 0 ? Optional.empty() : Optional.of(options.maxSize),
            options.maxAge.isZero() ? Optional.empty() : Optional.of(options.maxAge));

    var gc = new DiskCacheGarbageCollector(root, executorService, policy);

    CollectionStats stats = gc.run();

    System.out.println(stats.displayString());
    System.exit(0);
  }

  private static FileSystem getFileSystem() {
    // Note: the digest function is irrelevant, as the garbage collector scans the entire disk cache
    // and never computes digests.
    if (OS.getCurrent() == OS.WINDOWS) {
      return new WindowsFileSystem(DigestHashFunction.SHA256, false);
    }
    return new UnixFileSystem(DigestHashFunction.SHA256, "");
  }
}
