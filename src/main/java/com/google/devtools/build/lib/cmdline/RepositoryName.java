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

import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.LeafDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.LeafObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.LeafSerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Objects;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** The canonical name of an external repository. */
public final class RepositoryName {

  private static final Interner<RepositoryName> interner = BlazeInterners.newWeakInterner();

  @SerializationConstant
  public static final RepositoryName BAZEL_TOOLS = createUnvalidated("bazel_tools");

  @SerializationConstant public static final RepositoryName MAIN = createUnvalidated("");

  @SerializationConstant
  public static final RepositoryName BUILTINS = createUnvalidated("_builtins");

  private static final Pattern VALID_REPO_NAME = Pattern.compile("[\\w\\-.+]*");

  // Must start with a letter. Can contain ASCII letters and digits, underscore, dash, and dot.
  private static final Pattern VALID_USER_PROVIDED_NAME = Pattern.compile("[a-zA-Z0-9][-.\\w]*$");

  /**
   * A valid module name must: 1) begin with a lowercase letter; 2) end with a lowercase letter or a
   * digit; 3) contain only lowercase letters, digits, or one of * '._-'.
   */
  public static final Pattern VALID_MODULE_NAME = Pattern.compile("[a-z]([a-z0-9._-]*[a-z0-9])?");

  /**
   * Makes sure that name is a valid repository name and creates a new RepositoryName using it.
   *
   * @throws LabelSyntaxException if the name is invalid
   */
  public static RepositoryName create(String name) throws LabelSyntaxException {
    validate(name);
    return createUnvalidated(name);
  }

  /** Creates a RepositoryName from a known-valid string. */
  public static RepositoryName createUnvalidated(String name) {
    return interner.intern(new RepositoryName(name));
  }

  /**
   * Extracts the repository name from a PathFragment that was created with {@code
   * PackageIdentifier.getSourceRoot}.
   *
   * @return a {@code Pair} of the extracted repository name and the path fragment with the external
   *     repository prefix and repository name stripped, or null if none was found or the repository
   *     name was invalid.
   */
  @Nullable
  public static Pair<RepositoryName, PathFragment> fromPathFragment(
      PathFragment path, boolean siblingRepositoryLayout) {
    return fromPathFragment(
        path, siblingRepositoryLayout, /* bazelExternalDirectory= */ false);
  }

  @Nullable
  public static Pair<RepositoryName, PathFragment> fromPathFragment(
      PathFragment path, boolean siblingRepositoryLayout, boolean bazelExternalDirectory) {
    if (!path.isMultiSegment()) {
      return null;
    }

    PathFragment prefix =
        LabelConstants.getExternalPathPrefix(siblingRepositoryLayout, bazelExternalDirectory);
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
   * Store the name of the context repository where this repository name is requested. If this field
   * is not null, it means this instance represents the requested repository name that is actually
   * not visible from the context repository and should fail in {@code RepositoryDelegatorFunction}
   * when fetching the repository.
   */
  @Nullable private final RepositoryName contextRepoIfNotVisible;

  /**
   * If {@code contextRepoIfNotVisible} is not null, this field stores the suffix to be appended to
   * the error.
   */
  @Nullable private final String didYouMeanSuffix;

  private final int hashCode;

  private RepositoryName(
      String name,
      @Nullable RepositoryName contextRepoIfNotVisible,
      @Nullable String didYouMeanSuffix) {
    this.name = name;
    this.contextRepoIfNotVisible = contextRepoIfNotVisible;
    this.didYouMeanSuffix = didYouMeanSuffix;
    this.hashCode = HashCodes.hashObjects(name, contextRepoIfNotVisible, didYouMeanSuffix);
  }

  private RepositoryName(String name) {
    this(name, /* contextRepoIfNotVisible= */ null, /* didYouMeanSuffix= */ null);
  }

  /**
   * Performs validity checking, throwing an exception if the given name is invalid. The exception
   * message is sanitized.
   */
  static void validate(String name) throws LabelSyntaxException {
    if (name.isEmpty() || name.equals(BUILTINS.name)) {
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
              + " '_', '.', and must start with a letter or a number",
          StringUtilities.sanitizeControlChars(name));
    }
  }

  /** Returns true if the given name cannot possibly be a canonical repository name. */
  public static boolean isApparent(String name) {
    return !name.isEmpty() && !name.contains("+");
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
   * actually not visible from the context repository and should fail in {@code
   * RepositoryDelegatorFunction} when fetching with this {@link RepositoryName} instance.
   */
  public RepositoryName toNonVisible(RepositoryName contextRepo, String didYouMeanSuffix) {
    Preconditions.checkNotNull(contextRepo);
    Preconditions.checkArgument(contextRepo.isVisible());
    Preconditions.checkNotNull(didYouMeanSuffix);
    return new RepositoryName(name, contextRepo, didYouMeanSuffix);
  }

  @VisibleForTesting
  public RepositoryName toNonVisible(RepositoryName contextRepo) {
    return toNonVisible(contextRepo, "");
  }

  public boolean isVisible() {
    return contextRepoIfNotVisible == null;
  }

  public boolean isContextRepoMainRepo() {
    return !isVisible() && contextRepoIfNotVisible.isMain();
  }

  // Must only be called if isVisible() returns true.
  public String getContextRepoDisplayString() {
    Preconditions.checkNotNull(contextRepoIfNotVisible);
    if (contextRepoIfNotVisible.isMain()) {
      return "main repository";
    } else {
      return String.format("repository '%s'", contextRepoIfNotVisible);
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
      return String.format(
          "@@[unknown repo '%s' requested from %s%s]",
          name, contextRepoIfNotVisible, didYouMeanSuffix);
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
   *       <dd>if this repository is a direct dependency of the main module and its apparent name is
   *           "protobuf" (only if mainRepositoryMapping is not null)
   *       <dt><code>@@protobuf+</code>
   *       <dd>if this a repository that is not visible from the main module
   */
  public String getDisplayForm(@Nullable RepositoryMapping mainRepositoryMapping) {
    Preconditions.checkArgument(
        mainRepositoryMapping == null || mainRepositoryMapping.contextRepo().isMain());
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
   * __main__). Otherwise, the prefix is "external" by default and "bazel-external" when
   * --incompatible_bazel_external_directory is enabled.
   */
  public PathFragment getExecPath(boolean siblingRepositoryLayout) {
    return getExecPath(siblingRepositoryLayout, /* bazelExternalDirectory= */ false);
  }

  /** Returns the runfiles/execroot path for this repository under the requested layout. */
  public PathFragment getExecPath(
      boolean siblingRepositoryLayout, boolean bazelExternalDirectory) {
    if (isMain()) {
      return PathFragment.EMPTY_FRAGMENT;
    }
    PathFragment prefix =
        LabelConstants.getExternalPathPrefix(siblingRepositoryLayout, bazelExternalDirectory);
    return prefix.getRelative(getName());
  }

  /** Returns the runfiles path relative to the x.runfiles/main-repo directory. */
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
    if (!(object instanceof RepositoryName other)) {
      return false;
    }
    return Objects.equals(name, other.name)
        && Objects.equals(contextRepoIfNotVisible, other.contextRepoIfNotVisible)
        && Objects.equals(didYouMeanSuffix, other.didYouMeanSuffix);
  }

  @Override
  public int hashCode() {
    return hashCode;
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
      context.serializeLeaf(obj.contextRepoIfNotVisible, this, codedOut);
      context.serializeLeaf(obj.didYouMeanSuffix, stringCodec(), codedOut);
    }

    @Override
    public RepositoryName deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      String name = context.deserializeLeaf(codedIn, stringCodec());
      RepositoryName contextRepoIfNotVisible = context.deserializeLeaf(codedIn, this);
      String didYouMeanSuffix = context.deserializeLeaf(codedIn, stringCodec());
      RepositoryName repositoryName = createUnvalidated(name);
      return contextRepoIfNotVisible == null
          ? repositoryName
          : repositoryName.toNonVisible(contextRepoIfNotVisible, didYouMeanSuffix);
    }
  }
}
