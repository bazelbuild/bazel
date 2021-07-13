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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.regex.Pattern;

/** An immutable set of package name prefixes that should be ignored. */
@AutoCodec
public class IgnoredPackagePrefixesValue implements SkyValue {
  private final Set<PathFragment> ignoredPackagePrefixes;
  private final ImmutableSet<PathFragment> bazelignoreEntries;
  private final ImmutableSet<String> wildcardPatterns;
  private final Set<PathFragment> visitedPackage;
  private final Root workspaceRoot;
  private final Map<String, Pattern> patternCache;

  @AutoCodec @AutoCodec.VisibleForSerialization
  public static final IgnoredPackagePrefixesValue EMPTY_LIST =
          new IgnoredPackagePrefixesValue(ImmutableSet.of(), null);

  private IgnoredPackagePrefixesValue(ImmutableSet<PathFragment> patterns, Root workspaceRoot) {
    bazelignoreEntries = Preconditions.checkNotNull(patterns);

    ImmutableSet.Builder<String> wildcardPatternsBuilder = ImmutableSet.builder();
    ignoredPackagePrefixes = Sets.newConcurrentHashSet();

    for (PathFragment pattern: patterns) {
      String path = pattern.getPathString();
      if (UnixGlob.isWildcardFree(path)) {
        ignoredPackagePrefixes.add(pattern);
      } else {
        wildcardPatternsBuilder.add(path);
      }
    }

    wildcardPatterns = wildcardPatternsBuilder.build();
    patternCache = Maps.newHashMap();
    visitedPackage = Sets.newHashSet();
    this.workspaceRoot = workspaceRoot;
  }

  @AutoCodec.Instantiator
  public static IgnoredPackagePrefixesValue of(ImmutableSet<PathFragment> patterns, Root workspaceRoot) {
    return patterns.isEmpty() ? EMPTY_LIST : new IgnoredPackagePrefixesValue(patterns, workspaceRoot);
  }

  /** Creates a key from the main repository. */
  public static SkyKey key() {
    return Key.create(RepositoryName.MAIN);
  }

  /** Creates a key from the given repository name. */
  public static SkyKey key(RepositoryName repository) {
    return Key.create(repository);
  }

  public ImmutableSet<PathFragment> getPatterns() {
    return ImmutableSet.copyOf(ignoredPackagePrefixes);
  }

  public ImmutableSet<PathFragment> getPatterns(PathFragment base) {
    if (workspaceRoot == null || visitedPackage.contains(base)) {
      return ImmutableSet.copyOf(ignoredPackagePrefixes);
    }
    Path root = RootedPath.toRootedPath(workspaceRoot, base).asPath();
    visitedPackage.add(base);
    for (String pattern: wildcardPatterns) {
      if (UnixGlob.matches(pattern, base.getPathString(), patternCache)) {
        ignoredPackagePrefixes.add(base);
        return ImmutableSet.copyOf(ignoredPackagePrefixes);
      }
    }
    collectIgnoredDirectories(root);
    return ImmutableSet.copyOf(ignoredPackagePrefixes);
  }

  @Override
  public int hashCode() {
    return bazelignoreEntries.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof IgnoredPackagePrefixesValue) {
      IgnoredPackagePrefixesValue other = (IgnoredPackagePrefixesValue) obj;
      return this.bazelignoreEntries.equals(other.bazelignoreEntries);
    }
    return false;
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<RepositoryName> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(RepositoryName arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(RepositoryName arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.IGNORED_PACKAGE_PREFIXES;
    }
  }

  /**
   * Scan directories iteratively under {@code rootDirectory} and match them to avaliable patterns
   * in {@code wildcardPatterns}. Add the matched directories to {@code ignoredPackagePrefixesBuilder}.
   *
   * @param rootDirectory the directory of package
   */
  private void collectIgnoredDirectories(Path rootDirectory) {
    Stack<Path> rootStack = new Stack<>();
    rootStack.push(rootDirectory);
    Stack<String> relativeStack = new Stack<>();
    relativeStack.push("");

    while (!rootStack.isEmpty()) {
      Path root = rootStack.pop();
      String relative = relativeStack.pop();
      Collection<Dirent> dirents;
      try {
        dirents = root.readdir(Symlinks.FOLLOW);
      } catch (Exception ignored) {
        continue;
      }

      nextMatch:
      for (Dirent dirent : dirents) {
        String childRelative = relative.isEmpty()
                ? dirent.getName()
                : relative + "/" + dirent.getName();
        if (dirent.getType() == Dirent.Type.DIRECTORY) {
          for (String pattern: wildcardPatterns) {
            if (UnixGlob.matches(pattern, childRelative, patternCache)) {
              ignoredPackagePrefixes.add(PathFragment.create(childRelative));
              continue nextMatch;
            }
          }
          rootStack.push(root.getChild(dirent.getName()));
          relativeStack.push(childRelative);
        }
      }
    }
  }
}
