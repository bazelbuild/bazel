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
package com.google.devtools.build.lib.cmdline;

import com.google.common.base.Preconditions;
import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.cmdline.LabelValidator.BadLabelException;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Arrays;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/**
 * A class to identify a BUILD target. All targets belong to exactly one package. The name of a
 * target is called its label. A typical label looks like this: //dir1/dir2:target_name where
 * 'dir1/dir2' identifies the package containing a BUILD file, and 'target_name' identifies the
 * target within the package.
 *
 * <p>Parsing is robust against bad input, for example, from the command line.
 */
@StarlarkBuiltin(name = "Label", category = DocCategory.BUILTIN, doc = "A BUILD target identifier.")
@AutoCodec
@Immutable
@ThreadSafe
public final class Label
    implements Comparable<Label>, Serializable, StarlarkValue, SkyKey, CommandLineItem {

  /**
   * Package names that aren't made relative to the current repository because they mean special
   * things to Bazel.
   */
  public static final ImmutableSet<PathFragment> ABSOLUTE_PACKAGE_NAMES =
      ImmutableSet.of(
          // Used for select
          PathFragment.create("conditions"),
          // Visibility is labels aren't actually targets
          PathFragment.create("visibility"),
          // There is only one //external package
          LabelConstants.EXTERNAL_PACKAGE_NAME);

  public static final SkyFunctionName TRANSITIVE_TRAVERSAL =
      SkyFunctionName.createHermetic("TRANSITIVE_TRAVERSAL");

  private static final Interner<Label> LABEL_INTERNER = BlazeInterners.newWeakInterner();

  /**
   * Factory for Labels from absolute string form. e.g.
   *
   * <pre>
   * //foo/bar
   * //foo/bar:quux
   * {@literal @}foo
   * {@literal @}foo//bar
   * {@literal @}foo//bar:baz
   * </pre>
   *
   * <p>Treats labels in the default repository as being in the main repository instead.
   *
   * <p>Labels that begin with a repository name may have the repository name remapped to a
   * different name if it appears in {@code repositoryMapping}. This happens if the current
   * repository being evaluated is external to the main repository and the main repository set the
   * {@code repo_mapping} attribute when declaring this repository.
   *
   * @param absName label-like string to be parsed
   * @param repositoryMapping map of repository names from the local name found in the current
   *     repository to the global name declared in the main repository
   */
  public static Label parseAbsolute(
      String absName, ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws LabelSyntaxException {
    return parseAbsolute(
        absName, /* defaultToMain= */ true, repositoryMapping);
  }

  /**
   * Factory for Labels from absolute string form. e.g.
   *
   * <pre>
   * //foo/bar
   * //foo/bar:quux
   * {@literal @}foo
   * {@literal @}foo//bar
   * {@literal @}foo//bar:baz
   * </pre>
   *
   * <p>Labels that begin with a repository name may have the repository name remapped to a
   * different name if it appears in {@code repositoryMapping}. This happens if the current
   * repository being evaluated is external to the main repository and the main repository set the
   * {@code repo_mapping} attribute when declaring this repository.
   *
   * @param absName label-like string to be parsed
   * @param defaultToMain Treat labels in the default repository as being in the main one instead.
   * @param repositoryMapping map of repository names from the local name found in the current
   *     repository to the global name declared in the main repository
   */
  public static Label parseAbsolute(
      String absName,
      boolean defaultToMain,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws LabelSyntaxException {
    Preconditions.checkNotNull(repositoryMapping);
    String repo = defaultToMain ? "@" : RepositoryName.DEFAULT_REPOSITORY;
    int packageStartPos = absName.indexOf("//");
    if (packageStartPos > 0) {
      repo = absName.substring(0, packageStartPos);
      absName = absName.substring(packageStartPos);
    } else if (absName.startsWith("@")) {
      repo = absName;
      absName = "//:" + absName.substring(1);
    }
    String error = RepositoryName.validate(repo);
    if (error != null) {
      throw new LabelSyntaxException(
          "invalid repository name '" + StringUtilities.sanitizeControlChars(repo) + "': " + error);
    }
    try {
      LabelValidator.PackageAndTarget labelParts = LabelValidator.parseAbsoluteLabel(absName);
      PackageIdentifier pkgId =
          validatePackageName(
              labelParts.getPackageName(), labelParts.getTargetName(), repo, repositoryMapping);
      PathFragment packageFragment = pkgId.getPackageFragment();
      if (repo.isEmpty() && ABSOLUTE_PACKAGE_NAMES.contains(packageFragment)) {
        pkgId =
            PackageIdentifier.create(getGlobalRepoName("@", repositoryMapping), packageFragment);
      }
      return create(pkgId, labelParts.getTargetName());
    } catch (BadLabelException e) {
      throw new LabelSyntaxException(e.getMessage());
    }
  }

  private static RepositoryName getGlobalRepoName(
      String repo, ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws LabelSyntaxException {
    Preconditions.checkNotNull(repositoryMapping);
    RepositoryName repoName = RepositoryName.create(repo);
    return repositoryMapping.getOrDefault(repoName, repoName);
  }

  /**
   * Alternate factory method for Labels from absolute strings. This is a convenience method for
   * cases when a Label needs to be initialized statically, so the declared exception is
   * inconvenient.
   *
   * <p>Do not use this when the argument is not hard-wired.
   */
  @Deprecated
  // TODO(b/110698008): create parseAbsoluteUnchecked that passes repositoryMapping
  public static Label parseAbsoluteUnchecked(String absName, boolean defaultToMain) {
    try {
      return parseAbsolute(absName, defaultToMain, /* repositoryMapping= */ ImmutableMap.of());
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(e);
    }
  }

  public static Label parseAbsoluteUnchecked(String absName) {
    return parseAbsoluteUnchecked(absName, true);
  }

  /**
   * Factory for Labels from separate components.
   *
   * @param packageName The name of the package. The package name does <b>not</b> include {@code
   *     //}. Must be valid according to {@link LabelValidator#validatePackageName}.
   * @param targetName The name of the target within the package. Must be valid according to {@link
   *     LabelValidator#validateTargetName}.
   * @throws LabelSyntaxException if either of the arguments was invalid.
   */
  public static Label create(String packageName, String targetName) throws LabelSyntaxException {
    return create(validatePackageName(packageName, targetName), targetName);
  }

  /**
   * Similar factory to above, but takes a package identifier to allow external repository labels to
   * be created.
   */
  public static Label create(PackageIdentifier packageId, String targetName)
      throws LabelSyntaxException {
    return createUnvalidated(packageId, validateTargetName(packageId, targetName));
  }

  /**
   * Similar factory to above, but does not perform target name validation.
   *
   * <p>Only call this method if you know what you're doing; in particular, don't call it on
   * arbitrary {@code name} inputs
   */
  @AutoCodec.Instantiator
  public static Label createUnvalidated(PackageIdentifier packageIdentifier, String name) {
    return LABEL_INTERNER.intern(new Label(packageIdentifier, name));
  }

  /**
   * Resolves a relative label using a workspace-relative path to the current working directory. The
   * method handles these cases:
   *
   * <ul>
   *   <li>The label is absolute.
   *   <li>The label starts with a colon.
   *   <li>The label consists of a relative path, a colon, and a local part.
   *   <li>The label consists only of a local part.
   * </ul>
   *
   * <p>Note that this method does not support any of the special syntactic constructs otherwise
   * supported on the command line, like ":all", "/...", and so on.
   *
   * <p>It would be cleaner to use the TargetPatternEvaluator for this resolution, but that is not
   * possible, because it is sometimes necessary to resolve a relative label before the package path
   * is setup (maybe not anymore...)
   *
   * @throws LabelSyntaxException if the resulting label is not valid
   */
  public static Label parseCommandLineLabel(String label, PathFragment workspaceRelativePath)
      throws LabelSyntaxException {
    Preconditions.checkArgument(!workspaceRelativePath.isAbsolute());
    if (LabelValidator.isAbsolute(label)) {
      return parseAbsolute(label, ImmutableMap.of());
    }
    int index = label.indexOf(':');
    if (index < 0) {
      index = 0;
      label = ":" + label;
    }
    PathFragment path = workspaceRelativePath.getRelative(label.substring(0, index));
    // Use the String, String constructor, to make sure that the package name goes through the
    // validity check.
    return create(path.getPathString(), label.substring(index + 1));
  }

  /**
   * Validates the given target name and returns a normalized name if it is valid. Otherwise it
   * throws a SyntaxException.
   */
  private static String validateTargetName(PackageIdentifier packageIdentifier, String name)
      throws LabelSyntaxException {
    String error = LabelValidator.validateTargetName(name);
    if (error != null) {
      error = "invalid target name '" + StringUtilities.sanitizeControlChars(name) + "': " + error;
      if (packageIdentifier.getPackageFragment().getPathString().endsWith("/" + name)) {
        error += " (perhaps you meant \":" + name + "\"?)";
      }
      throw new LabelSyntaxException(error);
    }

    // TODO(bazel-team): This should be an error, but we can't make it one for legacy reasons.
    if (name.endsWith("/.")) {
      name = name.substring(0, name.length() - 2);
    }
    return name;
  }

  private static PackageIdentifier validatePackageName(String packageIdentifier, String name)
      throws LabelSyntaxException {
    return validatePackageName(
        packageIdentifier, name, /* repo= */ null, /* repositoryMapping= */ null);
  }

  /**
   * Validates the given package name and returns a canonical {@link PackageIdentifier} instance if
   * it is valid. Otherwise it throws a SyntaxException.
   */
  private static PackageIdentifier validatePackageName(
      String packageIdentifier,
      String name,
      String repo,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws LabelSyntaxException {
    String error = null;
    try {
      return PackageIdentifier.parse(packageIdentifier, repo, repositoryMapping);
    } catch (LabelSyntaxException e) {
      error = e.getMessage();
      error = "invalid package name '" + packageIdentifier + "': " + error;
      // This check is just for a more helpful error message
      // i.e. valid target name, invalid package name, colon-free label form
      // used => probably they meant "//foo:bar.c" not "//foo/bar.c".
      if (packageIdentifier.endsWith("/" + name)) {
        error += " (perhaps you meant \":" + name + "\"?)";
      }
      throw new LabelSyntaxException(error);
    }
  }

  /** The name and repository of the package. */
  private final PackageIdentifier packageIdentifier;

  /** The name of the target within the package. Canonical. */
  private final String name;

  private Label(PackageIdentifier packageIdentifier, String name) {
    Preconditions.checkNotNull(packageIdentifier);
    Preconditions.checkNotNull(name);

    this.packageIdentifier = packageIdentifier;
    this.name = name;
  }

  private Object writeReplace() {
    return new LabelSerializationProxy(getUnambiguousCanonicalForm());
  }

  private void readObject(ObjectInputStream unusedStream) throws InvalidObjectException {
    throw new InvalidObjectException("Serialization is allowed only by proxy");
  }

  public PackageIdentifier getPackageIdentifier() {
    return packageIdentifier;
  }

  public RepositoryName getRepository() {
    return packageIdentifier.getRepository();
  }

  /**
   * Returns the name of the package in which this rule was declared (e.g. {@code
   * //file/base:fileutils_test} returns {@code file/base}).
   */
  @StarlarkMethod(
      name = "package",
      structField = true,
      doc =
          "The package part of this label. "
              + "For instance:<br>"
              + "<pre class=language-python>Label(\"//pkg/foo:abc\").package == \"pkg/foo\"</pre>")
  public String getPackageName() {
    return packageIdentifier.getPackageFragment().getPathString();
  }

  /**
   * Returns the execution root for the workspace, relative to the execroot (e.g., for label
   * {@code @repo//pkg:b}, it will returns {@code external/repo/pkg} and for label {@code //pkg:a},
   * it will returns an empty string.
   *
   * @deprecated The sole purpose of this method is to implement the workspace_root method. For
   *     other purposes, use {@link RepositoryName#getExecPath} instead.
   */
  @StarlarkMethod(
      name = "workspace_root",
      structField = true,
      doc =
          "Returns the execution root for the workspace of this label, relative to the execroot. "
              + "For instance:<br>"
              + "<pre class=language-python>Label(\"@repo//pkg/foo:abc\").workspace_root =="
              + " \"external/repo\"</pre>",
      useStarlarkSemantics = true)
  @Deprecated
  public String getWorkspaceRootForStarlarkOnly(StarlarkSemantics semantics) {
    return packageIdentifier
        .getRepository()
        .getExecPath(semantics.getBool(BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT))
        .toString();
  }

  /**
   * Returns the path fragment of the package in which this rule was declared (e.g. {@code
   * //file/base:fileutils_test} returns {@code file/base}).
   *
   * <p>This is <b>not</b> suitable for inferring a path under which files related to a rule with
   * this label will be under the exec root, in particular, it won't work for rules in external
   * repositories.
   */
  public PathFragment getPackageFragment() {
    return packageIdentifier.getPackageFragment();
  }

  /**
   * Returns the label as a path fragment, using the package and the label name.
   *
   * <p>Make sure that the label refers to a file. Non-file labels do not necessarily have
   * PathFragment representations.
   */
  public PathFragment toPathFragment() {
    // PathFragments are normalized, so if we do this on a non-file target named '.'
    // then the package would be returned. Detect this and throw.
    // A target named '.' can never refer to a file.
    Preconditions.checkArgument(!name.equals("."));
    return packageIdentifier.getPackageFragment().getRelative(name);
  }

  /**
   * Returns the name by which this rule was declared (e.g. {@code //foo/bar:baz} returns {@code
   * baz}).
   */
  @StarlarkMethod(
      name = "name",
      structField = true,
      doc =
          "The name of this label within the package. "
              + "For instance:<br>"
              + "<pre class=language-python>Label(\"//pkg/foo:abc\").name == \"abc\"</pre>")
  public String getName() {
    return name;
  }

  /**
   * Renders this label in canonical form.
   *
   * <p>invariant: {@code parseAbsolute(x.toString(), false).equals(x)}
   */
  @Override
  public String toString() {
    return getCanonicalForm();
  }

  /**
   * Renders this label in canonical form.
   *
   * <p>invariant: {@code parseAbsolute(x.getCanonicalForm(), false).equals(x)}
   */
  public String getCanonicalForm() {
    return packageIdentifier.getCanonicalForm() + ":" + name;
  }

  public String getUnambiguousCanonicalForm() {
    return packageIdentifier.getRepository()
        + "//"
        + packageIdentifier.getPackageFragment()
        + ":"
        + name;
  }

  /** Return the name of the repository label refers to without the leading `at` symbol. */
  @StarlarkMethod(
      name = "workspace_name",
      structField = true,
      doc =
          "The repository part of this label. For instance, "
              + "<pre class=language-python>Label(\"@foo//bar:baz\").workspace_name"
              + " == \"foo\"</pre>")
  public String getWorkspaceName() {
    return packageIdentifier.getRepository().strippedName();
  }

  /**
   * Renders this label in shorthand form.
   *
   * <p>Labels with canonical form {@code //foo/bar:bar} have the shorthand form {@code //foo/bar}.
   * All other labels have identical shorthand and canonical forms.
   */
  public String toShorthandString() {
    if (!getPackageFragment().getBaseName().equals(name)) {
      return toString();
    }
    String repository;
    if (packageIdentifier.getRepository().isMain()) {
      repository = "";
    } else {
      repository = packageIdentifier.getRepository().getName();
    }
    return repository + "//" + getPackageFragment();
  }

  /**
   * Returns a label in the same package as this label with the given target name.
   *
   * @throws LabelSyntaxException if {@code targetName} is not a valid target name
   */
  public Label getLocalTargetLabel(String targetName) throws LabelSyntaxException {
    return create(packageIdentifier, targetName);
  }

  /**
   * Resolves a relative or absolute label name. If given name is absolute, then this method calls
   * {@link #parseAbsolute}. Otherwise, it calls {@link #getLocalTargetLabel}.
   *
   * <p>For example: {@code :quux} relative to {@code //foo/bar:baz} is {@code //foo/bar:quux};
   * {@code //wiz:quux} relative to {@code //foo/bar:baz} is {@code //wiz:quux}.
   *
   * @param relName the relative label name; must be non-empty.
   * @param thread the Starlark thread, which must provide a thread-local {@code HasRepoMapping}.
   */
  @StarlarkMethod(
      name = "relative",
      doc =
          "Resolves a label that is either absolute (starts with <code>//</code>) or relative to "
              + "the current package. If this label is in a remote repository, the argument will "
              + "be resolved relative to that repository. If the argument contains a repository "
              + "name, the current label is ignored and the argument is returned as-is, except "
              + "that the repository name is rewritten if it is in the current repository mapping. "
              + "Reserved labels will also be returned as-is.<br>"
              + "For example:<br>"
              + "<pre class=language-python>\n"
              + "Label(\"//foo/bar:baz\").relative(\":quux\") == Label(\"//foo/bar:quux\")\n"
              + "Label(\"//foo/bar:baz\").relative(\"//wiz:quux\") == Label(\"//wiz:quux\")\n"
              + "Label(\"@repo//foo/bar:baz\").relative(\"//wiz:quux\") == "
              + "Label(\"@repo//wiz:quux\")\n"
              + "Label(\"@repo//foo/bar:baz\").relative(\"//visibility:public\") == "
              + "Label(\"//visibility:public\")\n"
              + "Label(\"@repo//foo/bar:baz\").relative(\"@other//wiz:quux\") == "
              + "Label(\"@other//wiz:quux\")\n"
              + "</pre>"
              + "<p>If the repository mapping passed in is <code>{'@other' : '@remapped'}</code>, "
              + "then the following remapping will take place:<br>"
              + "<pre class=language-python>\n"
              + "Label(\"@repo//foo/bar:baz\").relative(\"@other//wiz:quux\") == "
              + "Label(\"@remapped//wiz:quux\")\n"
              + "</pre>",
      parameters = {
        @Param(name = "relName", doc = "The label that will be resolved relative to this one.")
      },
      useStarlarkThread = true)
  public Label getRelative(String relName, StarlarkThread thread) throws LabelSyntaxException {
    HasRepoMapping hrm = thread.getThreadLocal(HasRepoMapping.class);
    return getRelativeWithRemapping(relName, hrm.getRepoMapping());
  }

  /**
   * An interface for retrieving a repository mapping.
   *
   * <p>This has only a single implementation, {@code BazelStarlarkContext}, but we can't mention
   * that type here because logically it belongs in Bazel, above this package.
   */
  public interface HasRepoMapping {
    ImmutableMap<RepositoryName, RepositoryName> getRepoMapping();
  }

  /**
   * Resolves a relative or absolute label name. If given name is absolute, then this method calls
   * {@link #parseAbsolute}. Otherwise, it calls {@link #getLocalTargetLabel}.
   *
   * <p>For example: {@code :quux} relative to {@code //foo/bar:baz} is {@code //foo/bar:quux};
   * {@code //wiz:quux} relative to {@code //foo/bar:baz} is {@code //wiz:quux};
   * {@code @repo//foo:bar} relative to anything will be {@code @repo//foo:bar} if {@code @repo} is
   * not in {@code repositoryMapping} but will be {@code @other_repo//foo:bar} if there is an entry
   * {@code @repo -> @other_repo} in {@code repositoryMapping}
   *
   * @param relName the relative label name; must be non-empty
   * @param repositoryMapping the map of local repository names in external repository to global
   *     repository names in main repo; can be empty, but not null
   */
  public Label getRelativeWithRemapping(
      String relName, ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws LabelSyntaxException {
    Preconditions.checkNotNull(repositoryMapping);
    if (relName.length() == 0) {
      throw new LabelSyntaxException("empty package-relative label");
    }

    if (LabelValidator.isAbsolute(relName)) {
      return resolveRepositoryRelative(parseAbsolute(relName, false, repositoryMapping));
    } else if (relName.equals(":")) {
      throw new LabelSyntaxException("':' is not a valid package-relative label");
    } else if (relName.charAt(0) == ':') {
      return getLocalTargetLabel(relName.substring(1));
    } else {
      return getLocalTargetLabel(relName);
    }
  }

  /**
   * Resolves the repository of a label in the context of another label.
   *
   * <p>This is necessary so that dependency edges in remote repositories do not need to explicitly
   * mention their repository name. Otherwise, referring to e.g. <code>//a:b</code> in a remote
   * repository would point back to the main repository, which is usually not what is intended.
   *
   * <p>The return value will not be in the default repository.
   */
  public Label resolveRepositoryRelative(Label relative) {
    if (packageIdentifier.getRepository().isDefault()
        || !relative.packageIdentifier.getRepository().isDefault()) {
      return relative;
    } else {
      try {
        return create(
            PackageIdentifier.create(
                packageIdentifier.getRepository(), relative.getPackageFragment()),
            relative.getName());
      } catch (LabelSyntaxException e) {
        // We are creating the new label from an existing one which is guaranteed to be valid, so
        // this can't happen
        throw new IllegalStateException(e);
      }
    }
  }

  @Override
  public SkyFunctionName functionName() {
    return TRANSITIVE_TRAVERSAL;
  }

  @Override
  public int hashCode() {
    return hashCode(name, packageIdentifier);
  }

  /** Two labels are equal iff both their name and their package name are equal. */
  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof Label)) {
      return false;
    }
    Label otherLabel = (Label) other;
    // Package identifiers are interned so we compare them first.
    return packageIdentifier.equals(otherLabel.packageIdentifier) && name.equals(otherLabel.name);
  }

  /**
   * Defines the order between labels.
   *
   * <p>Labels are ordered primarily by package name and secondarily by target name. Both components
   * are ordered lexicographically. Thus {@code //a:b/c} comes before {@code //a/b:a}, i.e. the
   * position of the colon is significant to the order.
   */
  @Override
  public int compareTo(Label other) {
    if (this == other) {
      return 0;
    }
    return ComparisonChain.start()
        .compare(packageIdentifier, other.packageIdentifier)
        .compare(name, other.name)
        .result();
  }

  /**
   * Returns a suitable string for the user-friendly representation of the Label. Works even if the
   * argument is null.
   */
  public static String print(@Nullable Label label) {
    return label == null ? "(unknown)" : label.toString();
  }

  /**
   * Returns a {@link PathFragment} corresponding to the directory in which {@code label} would
   * reside, if it were interpreted to be a path.
   */
  public static PathFragment getContainingDirectory(Label label) {
    PathFragment pkg = label.getPackageFragment();
    String name = label.getName();
    if (name.equals(".")) {
      return pkg;
    }
    if (PathFragment.isNormalizedRelativePath(name) && !PathFragment.containsSeparator(name)) {
      // Optimize for the common case of a label like '//pkg:target'.
      return pkg;
    }
    return pkg.getRelative(name).getParentDirectory();
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("Label(");
    printer.repr(getCanonicalForm());
    printer.append(")");
  }

  @Override
  public void str(Printer printer) {
    printer.append(getCanonicalForm());
  }

  @Override
  public String expandToCommandLine() {
    return getCanonicalForm();
  }

  /**
   * Specialization of {@link Arrays#hashCode()} that does not require constructing a 2-element
   * array.
   */
  private static final int hashCode(Object obj1, Object obj2) {
    int result = 31 + (obj1 == null ? 0 : obj1.hashCode());
    return 31 * result + (obj2 == null ? 0 : obj2.hashCode());
  }
}
