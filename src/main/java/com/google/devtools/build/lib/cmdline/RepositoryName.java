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

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;
import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.skyframe.serialization.LeafDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.LeafObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.LeafSerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.OsPathPolicy;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.CompletionException;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** The canonical name of an external repository. */
public final class RepositoryName {

  @SerializationConstant
  public static final RepositoryName BAZEL_TOOLS = new RepositoryName("bazel_tools");

  @SerializationConstant public static final RepositoryName MAIN = new RepositoryName("");

  private static final Pattern VALID_REPO_NAME = Pattern.compile("[\\w\\-.+]*");

  // Must start with a letter. Can contain ASCII letters and digits, underscore, dash, and dot.
  private static final Pattern VALID_USER_PROVIDED_NAME = Pattern.compile("[a-zA-Z][-.\\w]*$");

  /**
   * A valid module name must: 1) begin with a lowercase letter; 2) end with a lowercase letter or a
   * digit; 3) contain only lowercase letters, digits, or one of * '._-'.
   */
  public static final Pattern VALID_MODULE_NAME = Pattern.compile("[a-z]([a-z0-9._-]*[a-z0-9])?");

  private static final LoadingCache<String, RepositoryName> repositoryNameCache =
      Caffeine.newBuilder()
          .weakValues()
          .build(
              name -> {
                validate(name);
                return new RepositoryName(name.intern());
              });

  /**
   * Makes sure that name is a valid repository name and creates a new RepositoryName using it.
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
      throwIfInstanceOf(e.getCause(), LabelSyntaxException.class);
      throwIfUnchecked(e.getCause());
      throw e;
    }
  }

  /** Creates a RepositoryName from a known-valid string. */
  public static RepositoryName createUnvalidated(String name) {
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
  @Nullable
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
   * Store the name of the owner repository where this repository name is requested. If this field
   * is not null, it means this instance represents the requested repository name that is actually
   * not visible from the owner repository and should fail in {@code RepositoryDelegatorFunction}
   * when fetching the repository.
   */
  @Nullable private final RepositoryName ownerRepoIfNotVisible;

  private RepositoryName(String name, @Nullable RepositoryName ownerRepoIfNotVisible) {
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
          "invalid repository name '%s': repo names are not allowed to be '%s'", name, name);
    }

    if (!VALID_REPO_NAME.matcher(name).matches()) {
      throw LabelParser.syntaxErrorf(
          "invalid repository name '%s': repo names may contain only A-Z, a-z, 0-9, '-', '_', '.'"
              + " and '+'",
          StringUtilities.sanitizeControlChars(name));
    }
  }

  /**
   * Validates a repo name provided by the user. Such names have tighter restrictions; for example,
   * they can only start with a letter, and cannot contain a plus (+).
   */
  public static void validateUserProvidedRepoName(String name) throws EvalException {
    if (!VALID_USER_PROVIDED_NAME.matcher(name).matches()) {
      throw Starlark.errorf(
          "invalid user-provided repo name '%s': valid names may contain only A-Z, a-z, 0-9, '-',"
              + " '_', '.', and must start with a letter",
          StringUtilities.sanitizeControlChars(name));
    }
  }

  /** Returns the bare repository name without the leading "{@literal @}". */
  public String getName() {
    return name;
  }

  /** Returns the marker file name for this repository. */
  public String getMarkerFileName() {
    return "@" + name + ".marker";
  }

  /**
   * Create a {@link RepositoryName} instance that indicates the requested repository name is
   * actually not visible from the owner repository and should fail in {@code
   * RepositoryDelegatorFunction} when fetching with this {@link RepositoryName} instance.
   */
  public RepositoryName toNonVisible(RepositoryName ownerRepo) {
    Preconditions.checkNotNull(ownerRepo);
    Preconditions.checkArgument(ownerRepo.isVisible());
    return new RepositoryName(name, ownerRepo);
  }

  public boolean isVisible() {
    return ownerRepoIfNotVisible == null;
  }

  // Must only be called if isVisible() returns true.
  public String getOwnerRepoDisplayString() {
    Preconditions.checkNotNull(ownerRepoIfNotVisible);
    if (ownerRepoIfNotVisible.isMain()) {
      return "main repository";
    } else {
      return String.format("repository '%s'", ownerRepoIfNotVisible);
    }
  }

  // Must only be called if isVisible() returns true.
  public String getOwnerModuleDisplayString() {
    Preconditions.checkNotNull(ownerRepoIfNotVisible);
    if (ownerRepoIfNotVisible.isMain()) {
      return "root module";
    } else {
      return String.format(
          "module '%s'",
          ownerRepoIfNotVisible
              .getName()
              .substring(0, ownerRepoIfNotVisible.getName().indexOf('+')));
    }
  }

  /** Returns if this is the main repository. */
  public boolean isMain() {
    return equals(MAIN);
  }

  /**
   * Returns the repository name, with two leading "{@literal @}"s, indicating that this is a
   * canonical repo name.
   */
  // TODO(bazel-team): Rename to "getCanonicalForm".
  public String getNameWithAt() {
    if (!isVisible()) {
      return String.format("@@[unknown repo '%s' requested from %s]", name, ownerRepoIfNotVisible);
    }
    return "@@" + name;
  }

  /**
   * Returns the repository name with leading "{@literal @}"s except for the main repo, which is
   * just the empty string.
   */
  // TODO(bazel-team): Rename to "getDefaultForm".
  public String getCanonicalForm() {
    return isMain() ? "" : getNameWithAt();
  }

  /**
   * Returns the repository part of a {@link Label}'s string representation suitable for display.
   * The returned string is as simple as possible in the context of the main repo whose repository
   * mapping is provided: an empty string for the main repo, or a string prefixed with a leading
   * "{@literal @}" or "{@literal @@}" otherwise.
   *
   * @param mainRepositoryMapping the {@link RepositoryMapping} of the main repository
   * @return
   *     <dl>
   *       <dt>the empty string
   *       <dd>if this is the main repository
   *       <dt><code>@protobuf</code>
   *       <dd>if this repository is a WORKSPACE dependency and its <code>name</code> is "protobuf",
   *           or if this repository is a Bzlmod dependency of the main module and its apparent name
   *           is "protobuf" (in both cases only if mainRepositoryMapping is not null)
   *       <dt><code>@@protobuf+</code>
   *       <dd>only with Bzlmod, if this a repository that is not visible from the main module
   */
  public String getDisplayForm(@Nullable RepositoryMapping mainRepositoryMapping) {
    Preconditions.checkArgument(
        mainRepositoryMapping == null
            || mainRepositoryMapping.ownerRepo() == null
            || mainRepositoryMapping.ownerRepo().isMain());
    if (!isVisible()) {
      return getNameWithAt();
    }
    if (isMain()) {
      // Packages in the main repository can always use repo-relative form.
      return "";
    }
    if (mainRepositoryMapping == null) {
      return getNameWithAt();
    }
    if (!mainRepositoryMapping.usesStrictDeps()) {
      // If the main repository mapping is not using strict visibility, then Bzlmod is certainly
      // disabled, which means that canonical and apparent names can be used interchangeably from
      // the context of the main repository.
      return '@' + getName();
    }
    // If possible, represent the repository with a non-canonical label using the apparent name the
    // main repository has for it, otherwise fall back to a canonical label.
    return mainRepositoryMapping
        .getInverse(this)
        .map(apparentName -> "@" + apparentName)
        .orElse(getNameWithAt());
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

  /** Same as {@link #getNameWithAt}. */
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
        && Objects.equals(ownerRepoIfNotVisible, other.ownerRepoIfNotVisible);
  }

  @Override
  public int hashCode() {
    return Objects.hash(OsPathPolicy.getFilePathOs().hash(name), ownerRepoIfNotVisible);
  }

  public static Codec repositoryNameCodec() {
    return Codec.INSTANCE;
  }

  private static final class Codec extends LeafObjectCodec<RepositoryName> {
    private static final Codec INSTANCE = new Codec();

    @Override
    public Class<RepositoryName> getEncodedClass() {
      return RepositoryName.class;
    }

    @Override
    public void serialize(
        LeafSerializationContext context, RepositoryName obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serializeLeaf(obj.getName(), stringCodec(), codedOut);
      context.serializeLeaf(obj.ownerRepoIfNotVisible, this, codedOut);
    }

    @Override
    public RepositoryName deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return new RepositoryName(
          context.deserializeLeaf(codedIn, stringCodec()), context.deserializeLeaf(codedIn, this));
    }
  }
}
