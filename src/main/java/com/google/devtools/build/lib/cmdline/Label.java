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

import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.LabelValidator.BadLabelException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrintableValue;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;

/**
 * A class to identify a BUILD target. All targets belong to exactly one package.
 * The name of a target is called its label. A typical label looks like this:
 * //dir1/dir2:target_name where 'dir1/dir2' identifies the package containing a BUILD file,
 * and 'target_name' identifies the target within the package.
 *
 * <p>Parsing is robust against bad input, for example, from the command line.
 */
@SkylarkModule(name = "Label", doc = "A BUILD target identifier.")
@Immutable @ThreadSafe
public final class Label implements Comparable<Label>, Serializable, SkylarkPrintableValue {
  public static final PathFragment EXTERNAL_PACKAGE_NAME = new PathFragment("external");

  /**
   * Package names that aren't made relative to the current repository because they mean special
   * things to Bazel.
   */
  private static final ImmutableSet<PathFragment> ABSOLUTE_PACKAGE_NAMES = ImmutableSet.of(
      // dependencies that are a function of the configuration
      new PathFragment("tools/defaults"),
      // Visibility is labels aren't actually targets
      new PathFragment("visibility"),
      // There is only one //external package
      Label.EXTERNAL_PACKAGE_NAME);

  public static final PackageIdentifier EXTERNAL_PACKAGE_IDENTIFIER =
      PackageIdentifier.createInDefaultRepo(EXTERNAL_PACKAGE_NAME);

  public static final String EXTERNAL_PATH_PREFIX = "external";

  /**
   * Factory for Labels from absolute string form. e.g.
   * <pre>
   * //foo/bar
   * //foo/bar:quux
   * {@literal @}foo//bar
   * {@literal @}foo//bar:baz
   * </pre>
   */
  public static Label parseAbsolute(String absName) throws LabelSyntaxException {
    String repo = PackageIdentifier.DEFAULT_REPOSITORY;
    int packageStartPos = absName.indexOf("//");
    if (packageStartPos > 0) {
      repo = absName.substring(0, packageStartPos);
      absName = absName.substring(packageStartPos);
    }
    try {
      LabelValidator.PackageAndTarget labelParts = LabelValidator.parseAbsoluteLabel(absName);
      PackageIdentifier pkgIdWithoutRepo =
          validate(labelParts.getPackageName(), labelParts.getTargetName());
      return new Label(
          PackageIdentifier.create(repo, pkgIdWithoutRepo.getPackageFragment()),
          labelParts.getTargetName());
    } catch (BadLabelException e) {
      throw new LabelSyntaxException(e.getMessage());
    }
  }

  /**
   * Alternate factory method for Labels from absolute strings. This is a convenience method for
   * cases when a Label needs to be initialized statically, so the declared exception is
   * inconvenient.
   *
   * <p>Do not use this when the argument is not hard-wired.
   */
  public static Label parseAbsoluteUnchecked(String absName) {
    try {
      return parseAbsolute(absName);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(e);
    }
  }

  /**
   * Factory for Labels from separate components.
   *
   * @param packageName The name of the package.  The package name does
   *   <b>not</b> include {@code //}.  Must be valid according to
   *   {@link LabelValidator#validatePackageName}.
   * @param targetName The name of the target within the package.  Must be
   *   valid according to {@link LabelValidator#validateTargetName}.
   * @throws LabelSyntaxException if either of the arguments was invalid.
   */
  public static Label create(String packageName, String targetName) throws LabelSyntaxException {
    return new Label(packageName, targetName);
  }

  /**
   * Similar factory to above, but takes a package identifier to allow external repository labels
   * to be created.
   */
  public static Label create(PackageIdentifier packageId, String targetName)
      throws LabelSyntaxException {
    return new Label(packageId, targetName);
  }

  /**
   * Resolves a relative label using a workspace-relative path to the current working directory. The
   * method handles these cases:
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
   * is setup; in particular, before the tools/defaults package is created.
   *
   * @throws LabelSyntaxException if the resulting label is not valid
   */
  public static Label parseCommandLineLabel(String label, PathFragment workspaceRelativePath)
      throws LabelSyntaxException {
    Preconditions.checkArgument(!workspaceRelativePath.isAbsolute());
    if (LabelValidator.isAbsolute(label)) {
      return parseAbsolute(label);
    }
    int index = label.indexOf(':');
    if (index < 0) {
      index = 0;
      label = ":" + label;
    }
    PathFragment path = workspaceRelativePath.getRelative(label.substring(0, index));
    // Use the String, String constructor, to make sure that the package name goes through the
    // validity check.
    return new Label(path.getPathString(), label.substring(index + 1));
  }

  /**
   * Validates the given target name and returns a canonical String instance if it is valid.
   * Otherwise it throws a SyntaxException.
   */
  private static String canonicalizeTargetName(String name) throws LabelSyntaxException {
    String error = LabelValidator.validateTargetName(name);
    if (error != null) {
      error = "invalid target name '" + StringUtilities.sanitizeControlChars(name) + "': " + error;
      throw new LabelSyntaxException(error);
    }

    // TODO(bazel-team): This should be an error, but we can't make it one for legacy reasons.
    if (name.endsWith("/.")) {
      name = name.substring(0, name.length() - 2);
    }

    return StringCanonicalizer.intern(name);
  }

  /**
   * Validates the given package name and returns a canonical {@link PackageIdentifier} instance
   * if it is valid. Otherwise it throws a SyntaxException.
   */
  private static PackageIdentifier validate(String packageIdentifier, String name)
      throws LabelSyntaxException {
    String error = null;
    try {
      return PackageIdentifier.parse(packageIdentifier);
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

  /**
   * Constructor from a package name, target name. Both are checked for validity
   * and a SyntaxException is thrown if either is invalid.
   * TODO(bazel-team): move the validation to {@link PackageIdentifier}. Unfortunately, there are a
   * bazillion tests that use invalid package names (taking advantage of the fact that calling
   * Label(PathFragment, String) doesn't validate the package name).
   */
  private Label(String packageIdentifier, String name) throws LabelSyntaxException {
    this(validate(packageIdentifier, name), name);
  }

  private Label(PackageIdentifier packageIdentifier, String name)
      throws LabelSyntaxException {
    Preconditions.checkNotNull(packageIdentifier);
    Preconditions.checkNotNull(name);

    try {
      this.packageIdentifier = packageIdentifier;
      this.name = canonicalizeTargetName(name);
    } catch (LabelSyntaxException e) {
      // This check is just for a more helpful error message
      // i.e. valid target name, invalid package name, colon-free label form
      // used => probably they meant "//foo:bar.c" not "//foo/bar.c".
      if (packageIdentifier.getPackageFragment().getPathString().endsWith("/" + name)) {
        throw new LabelSyntaxException(e.getMessage() + " (perhaps you meant \":" + name + "\"?)");
      }
      throw e;
    }
  }

  private Object writeReplace() {
    return new LabelSerializationProxy(toString());
  }

  private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Serialization is allowed only by proxy");
  }

  public PackageIdentifier getPackageIdentifier() {
    return packageIdentifier;
  }

  /**
   * Returns the name of the package in which this rule was declared (e.g. {@code
   * //file/base:fileutils_test} returns {@code file/base}).
   */
  @SkylarkCallable(name = "package", structField = true,
      doc = "The package part of this label. "
      + "For instance:<br>"
      + "<pre class=language-python>Label(\"//pkg/foo:abc\").package == \"pkg/foo\"</pre>")
  public String getPackageName() {
    return packageIdentifier.getPackageFragment().getPathString();
  }
  
  /**
   * Returns the execution root for the workspace, relative to the execroot (e.g., for label
   * {@code @repo//pkg:b}, it will returns {@code external/repo/pkg} and for label {@code //pkg:a},
   * it will returns an empty string.
   */
  @SkylarkCallable(name = "workspace_root", structField = true,
      doc = "Returns the execution root for the workspace of this label, relative to the execroot. "
      + "For instance:<br>"
      + "<pre class=language-python>Label(\"@repo//pkg/foo:abc\").workspace_root =="
      + " \"external/repo\"</pre>")
  public String getWorkspaceRoot() {
    return packageIdentifier.getRepository().getPathFragment().toString();
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

  public static final com.google.common.base.Function<Label, PathFragment> PACKAGE_FRAGMENT =
      new com.google.common.base.Function<Label, PathFragment>() {
        @Override
        public PathFragment apply(Label label) {
          return label.getPackageFragment();
        }
  };

  /**
   * Returns the label as a path fragment, using the package and the label name.
   */
  public PathFragment toPathFragment() {
    return packageIdentifier.getPackageFragment().getRelative(name);
  }

  /**
   * Returns the name by which this rule was declared (e.g. {@code //foo/bar:baz}
   * returns {@code baz}).
   */
  @SkylarkCallable(name = "name", structField = true,
      doc = "The name of this label within the package. "
      + "For instance:<br>"
      + "<pre class=language-python>Label(\"//pkg/foo:abc\").name == \"abc\"</pre>")
  public String getName() {
    return name;
  }

  /**
   * Renders this label in canonical form.
   *
   * <p>invariant: {@code parseAbsolute(x.toString()).equals(x)}
   */
  @Override
  public String toString() {
    return getCanonicalForm();
  }

  /**
   * Renders this label in canonical form.
   *
   * <p>invariant: {@code parseAbsolute(x.toString()).equals(x)}
   */
  public String getCanonicalForm() {
    return packageIdentifier.getRepository() + "//" + packageIdentifier.getPackageFragment()
        + ":" + name;
  }

  /**
   * Renders this label in shorthand form.
   *
   * <p>Labels with canonical form {@code //foo/bar:bar} have the shorthand form {@code //foo/bar}.
   * All other labels have identical shorthand and canonical forms.
   */
  public String toShorthandString() {
    return packageIdentifier.getRepository() + (getPackageFragment().getBaseName().equals(name)
        ? "//" + getPackageFragment()
        : toString());
  }

  /**
   * Returns a label in the same package as this label with the given target name.
   *
   * @throws LabelSyntaxException if {@code targetName} is not a valid target name
   */
  public Label getLocalTargetLabel(String targetName) throws LabelSyntaxException {
    return new Label(packageIdentifier, targetName);
  }

  /**
   * Resolves a relative or absolute label name. If given name is absolute, then this method calls
   * {@link #parseAbsolute}. Otherwise, it calls {@link #getLocalTargetLabel}.
   *
   * <p>For example:
   * {@code :quux} relative to {@code //foo/bar:baz} is {@code //foo/bar:quux};
   * {@code //wiz:quux} relative to {@code //foo/bar:baz} is {@code //wiz:quux}.
   *
   * @param relName the relative label name; must be non-empty.
   */
  @SkylarkCallable(name = "relative", doc =
        "Resolves a label that is either absolute (starts with <code>//</code>) or relative to the"
      + " current package.<br>"
      + "For example:<br>"
      + "<pre class=language-python>\n"
      + "Label(\"//foo/bar:baz\").relative(\":quux\") == Label(\"//foo/bar:quux\")\n"
      + "Label(\"//foo/bar:baz\").relative(\"//wiz:quux\") == Label(\"//wiz:quux\")\n"
      + "</pre>")
  public Label getRelative(String relName) throws LabelSyntaxException {
    if (relName.length() == 0) {
      throw new LabelSyntaxException("empty package-relative label");
    }
    if (LabelValidator.isAbsolute(relName)) {
      return parseAbsolute(relName);
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
   */
  public Label resolveRepositoryRelative(Label relative) {
    if (relative.packageIdentifier.getRepository().getName().equals("@")) {
      try {
        return new Label(
            PackageIdentifier.create(
                PackageIdentifier.DEFAULT_REPOSITORY_NAME,
                relative.packageIdentifier.getPackageFragment()),
            relative.getName());
      } catch (LabelSyntaxException e) {
        throw new IllegalStateException(e);
      }
    }

    if (packageIdentifier.getRepository().isDefault()
        || !relative.packageIdentifier.getRepository().isDefault()
        || ABSOLUTE_PACKAGE_NAMES.contains(relative.getPackageIdentifier().getPackageFragment())) {
      return relative;
    } else {
      try {
        return new Label(
            PackageIdentifier
                .create(packageIdentifier.getRepository(), relative.getPackageFragment()),
            relative.getName());
      } catch (LabelSyntaxException e) {
        // We are creating the new label from an existing one which is guaranteed to be valid, so
        // this can't happen
        throw new IllegalStateException(e);
      }
    }
  }

  @Override
  public int hashCode() {
    return name.hashCode() ^ packageIdentifier.hashCode();
  }

  /**
   * Two labels are equal iff both their name and their package name are equal.
   */
  @Override
  public boolean equals(Object other) {
    if (!(other instanceof Label)) {
      return false;
    }
    Label otherLabel = (Label) other;
    return name.equals(otherLabel.name) // least likely one first
        && packageIdentifier.equals(otherLabel.packageIdentifier);
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
    return ComparisonChain.start()
        .compare(packageIdentifier, other.packageIdentifier)
        .compare(name, other.name)
        .result();
  }

  /**
   * Returns a suitable string for the user-friendly representation of the Label. Works even if the
   * argument is null.
   */
  public static String print(Label label) {
    return label == null ? "(unknown)" : label.toString();
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void write(Appendable buffer, char quotationMark) {
    // We don't use the Skylark Printer class here to avoid creating a circular dependency.
    //
    // TODO(bazel-team): make the representation readable Label(//foo),
    // and isolate the legacy functions that want the unreadable variant.
    try {
      // There is no need to escape the contents of the Label since characters that might otherwise
      // require escaping are disallowed.
      buffer.append(quotationMark);
      buffer.append(toString());
      buffer.append(quotationMark);
    } catch (IOException e) {
      // This function will only be used with in-memory Appendables, hence we should never get here.
      throw new AssertionError(e);
    }
  }

  @Override
  public void print(Appendable buffer, char quotationMark) {
    // We don't use the Skylark Printer class here to avoid creating a circular dependency.
    //
    // TODO(bazel-team): make the representation readable Label(//foo),
    // and isolate the legacy functions that want the unreadable variant.
    try {
      buffer.append(toString());
    } catch (IOException e) {
      // This function will only be used with in-memory Appendables, hence we should never get here.
      throw new AssertionError(e);
    }
  }
}
