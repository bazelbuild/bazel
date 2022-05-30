// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.OsPathPolicy;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import java.util.concurrent.CompletionException;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** The name of an external repository. */
public final class RepositoryName {

  @SerializationConstant
  public static final RepositoryName BAZEL_TOOLS = new RepositoryName("bazel_tools");

  @SerializationConstant public static final RepositoryName MAIN = new RepositoryName("");

  private static final Pattern VALID_REPO_NAME = Pattern.compile("[\\w\\-.]*");

  private static final LoadingCache<String, RepositoryName> repositoryNameCache =
      Caffeine.newBuilder()
          .weakValues()
          .build(
              name -> {
                validate(name);
                return new RepositoryName(StringCanonicalizer.intern(name));
              });

  /**
   * Makes sure that name is a valid repository name and creates a new RepositoryName using it. The
   * given string must not begin with a '@'.
   *
   * @throws LabelSyntaxException if the name is invalid
   */
  public static RepositoryName create(String name) throws LabelSyntaxException {
    if (name.isEmpty()) {
      return MAIN;
    }
    try {
      return repositoryNameCache.get(name);
    } catch (CompletionException e) {
      Throwables.propagateIfPossible(e.getCause(), LabelSyntaxException.class);
      throw e;
    }
  }

  /**
   * Creates a RepositoryName from a known-valid string. The given string must not begin with a '@'.
   */
  public static RepositoryName createUnvalidated(String name) {
    Preconditions.checkArgument(!name.startsWith("@"), "Do not prefix @ to repo names!");
    if (name.isEmpty()) {
      // NOTE(wyv): Without this `if` clause, a lot of Google-internal integration tests would start
      //   failing. This suggests to me that something is comparing RepositoryName objects using
      //   reference equality instead of #equals().
      return MAIN;
    }
    return repositoryNameCache.get(name);
  }

  /**
   * Extracts the repository name from a PathFragment that was created with {@code
   * PackageIdentifier.getSourceRoot}.
   *
   * @return a {@code Pair} of the extracted repository name and the path fragment with stripped of
   *     "external/"-prefix and repository name, or null if none was found or the repository name
   *     was invalid.
   */
  public static Pair<RepositoryName, PathFragment> fromPathFragment(
      PathFragment path, boolean siblingRepositoryLayout) {
    if (!path.isMultiSegment()) {
      return null;
    }

    PathFragment prefix =
        siblingRepositoryLayout
            ? LabelConstants.EXPERIMENTAL_EXTERNAL_PATH_PREFIX
            : LabelConstants.EXTERNAL_PATH_PREFIX;
    if (!path.startsWith(prefix)) {
      return null;
    }

    try {
      RepositoryName repoName = create(path.getSegment(1));
      PathFragment subPath = path.subFragment(2);
      return Pair.of(repoName, subPath);
    } catch (LabelSyntaxException e) {
      return null;
    }
  }

  private final String name;

  /**
   * Store the name if the owner repository where this repository name is requested. If this field
   * is not null, it means this instance represents the requested repository name that is actually
   * not visible from the owner repository and should fail in {@code RepositoryDelegatorFunction}
   * when fetching the repository.
   */
  private final String ownerRepoIfNotVisible;

  private RepositoryName(String name, String ownerRepoIfNotVisible) {
    this.name = name;
    this.ownerRepoIfNotVisible = ownerRepoIfNotVisible;
  }

  private RepositoryName(String name) {
    this(name, null);
  }

  /**
   * Performs validity checking, throwing an exception if the given name is invalid. The exception
   * message is sanitized.
   */
  static void validate(String name) throws LabelSyntaxException {
    if (name.isEmpty()) {
      return;
    }

    // Some special cases for more user-friendly error messages.
    if (name.equals(".") || name.equals("..")) {
      throw LabelParser.syntaxErrorf(
          "invalid repository name '@%s': repo names are not allowed to be '@%s'", name, name);
    }

    if (!VALID_REPO_NAME.matcher(name).matches()) {
      throw LabelParser.syntaxErrorf(
          "invalid repository name '@%s': repo names may contain only A-Z, a-z, 0-9, '-', '_' and"
              + " '.'",
          StringUtilities.sanitizeControlChars(name));
    }
  }

  /** Returns the bare repository name without the leading "{@literal @}". */
  public String getName() {
    return name;
  }

  /**
   * Create a {@link RepositoryName} instance that indicates the requested repository name is
   * actually not visible from the owner repository and should fail in {@code
   * RepositoryDelegatorFunction} when fetching with this {@link RepositoryName} instance.
   */
  public RepositoryName toNonVisible(String ownerRepo) {
    Preconditions.checkNotNull(ownerRepo);
    return new RepositoryName(name, ownerRepo);
  }

  public boolean isVisible() {
    return ownerRepoIfNotVisible == null;
  }

  @Nullable
  public String getOwnerRepoIfNotVisible() {
    return ownerRepoIfNotVisible;
  }

  /** Returns if this is the main repository, that is, {@link #getName} is empty. */
  public boolean isMain() {
    return name.isEmpty();
  }

  /** Returns the repository name, with leading "{@literal @}". */
  public String getNameWithAt() {
    return '@' + name;
  }

  /**
   * Returns the repository name with leading "{@literal @}" except for the main repo, which is just
   * the empty string.
   */
  // TODO(bazel-team): Consider renaming to "getDefaultForm".
  public String getCanonicalForm() {
    return isMain() ? "" : getNameWithAt();
  }

  /**
   * Returns the runfiles/execRoot path for this repository. If we don't know the name of this repo
   * (i.e., it is in the main repository), return an empty path fragment.
   *
   * <p>If --experimental_sibling_repository_layout is true, return "$execroot/../repo" (sibling of
   * __main__), instead of "$execroot/external/repo".
   */
  public PathFragment getExecPath(boolean siblingRepositoryLayout) {
    if (isMain()) {
      return PathFragment.EMPTY_FRAGMENT;
    }
    PathFragment prefix =
        siblingRepositoryLayout
            ? LabelConstants.EXPERIMENTAL_EXTERNAL_PATH_PREFIX
            : LabelConstants.EXTERNAL_PATH_PREFIX;
    return prefix.getRelative(getName());
  }

  /**
   * Returns the runfiles path relative to the x.runfiles/main-repo directory.
   */
  // TODO(kchodorow): remove once execroot is reorg-ed.
  public PathFragment getRunfilesPath() {
    return isMain()
        ? PathFragment.EMPTY_FRAGMENT
        : PathFragment.create("..").getRelative(getName());
  }

  /** Returns the repository name, with leading "{@literal @}". */
  @Override
  public String toString() {
    return getNameWithAt();
  }

  @Override
  public boolean equals(Object object) {
    if (this == object) {
      return true;
    }
    if (!(object instanceof RepositoryName)) {
      return false;
    }
    RepositoryName other = (RepositoryName) object;
    return OsPathPolicy.getFilePathOs().equals(name, other.name)
        && OsPathPolicy.getFilePathOs().equals(ownerRepoIfNotVisible, other.ownerRepoIfNotVisible);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        OsPathPolicy.getFilePathOs().hash(name),
        OsPathPolicy.getFilePathOs().hash(ownerRepoIfNotVisible));
  }
}
