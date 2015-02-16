// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.ComparisonChain;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.LabelValidator.BadLabelException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;

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
public final class Label implements Comparable<Label>, Serializable {

  /**
   * Thrown by the parsing methods to indicate a bad label.
   */
  public static class SyntaxException extends Exception {
    public SyntaxException(String message) {
      super(message);
    }
  }

  /**
   * Factory for Labels from absolute string form, possibly including a repository name prefix. For
   * example:
   * <pre>
   * //foo/bar
   * {@literal @}foo//bar
   * {@literal @}foo//bar:baz
   * </pre>
   */
  public static Label parseRepositoryLabel(String absName) throws SyntaxException {
    String repo = PackageIdentifier.DEFAULT_REPOSITORY;
    int packageStartPos = absName.indexOf("//");
    if (packageStartPos > 0) {
      repo = absName.substring(0, packageStartPos);
      absName = absName.substring(packageStartPos);
    }
    try {
      LabelValidator.PackageAndTarget labelParts = LabelValidator.parseAbsoluteLabel(absName);
      return new Label(new PackageIdentifier(repo, new PathFragment(labelParts.getPackageName())),
          labelParts.getTargetName());
    } catch (BadLabelException e) {
      throw new SyntaxException(e.getMessage());
    }
  }

  /**
   * Factory for Labels from absolute string form. e.g.
   * <pre>
   * //foo/bar
   * //foo/bar:quux
   * </pre>
   */
  public static Label parseAbsolute(String absName) throws SyntaxException {
    try {
      LabelValidator.PackageAndTarget labelParts = LabelValidator.parseAbsoluteLabel(absName);
      return create(labelParts.getPackageName(), labelParts.getTargetName());
    } catch (BadLabelException e) {
      throw new SyntaxException(e.getMessage());
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
    } catch (SyntaxException e) {
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
   * @throws SyntaxException if either of the arguments was invalid.
   */
  public static Label create(String packageName, String targetName) throws SyntaxException {
    return new Label(packageName, targetName);
  }

  /**
   * Similar factory to above, but takes a package identifier to allow external repository labels
   * to be created.
   */
  public static Label create(PackageIdentifier packageId, String targetName)
      throws SyntaxException {
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
   * @throws SyntaxException if the resulting label is not valid
   */
  public static Label parseCommandLineLabel(String label, PathFragment workspaceRelativePath)
      throws SyntaxException {
    Preconditions.checkArgument(!workspaceRelativePath.isAbsolute());
    if (label.startsWith("//")) {
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
  private static String canonicalizeTargetName(String name) throws SyntaxException {
    String error = LabelValidator.validateTargetName(name);
    if (error != null) {
      error = "invalid target name '" + StringUtilities.sanitizeControlChars(name) + "': " + error;
      throw new SyntaxException(error);
    }

    // TODO(bazel-team): This should be an error, but we can't make it one for legacy reasons.
    if (name.endsWith("/.")) {
      name = name.substring(0, name.length() - 2);
    }

    return StringCanonicalizer.intern(name);
  }

  /**
   * Validates the given package name and returns a canonical PathFragment instance if it is valid.
   * Otherwise it throws a SyntaxException.
   */
  private static PathFragment validate(String packageName, String name) throws SyntaxException {
    String error = LabelValidator.validatePackageName(packageName);
    if (error != null) {
      error = "invalid package name '" + packageName + "': " + error;
      // This check is just for a more helpful error message
      // i.e. valid target name, invalid package name, colon-free label form
      // used => probably they meant "//foo:bar.c" not "//foo/bar.c".
      if (packageName.endsWith("/" + name)) {
        error += " (perhaps you meant \":" + name + "\"?)";
      }
      throw new SyntaxException(error);
    }
    return new PathFragment(packageName);
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
  private Label(String packageName, String name) throws SyntaxException {
    this(validate(packageName, name), name);
  }

  /**
   * Constructor from canonical valid package name and a target name. The target
   * name is checked for validity and a SyntaxException is throw if it isn't.
   */
  private Label(PathFragment packageName, String name) throws SyntaxException {
    this(PackageIdentifier.createInDefaultRepo(packageName), name);
  }

  private Label(PackageIdentifier packageIdentifier, String name)
      throws SyntaxException {
    Preconditions.checkNotNull(packageIdentifier);
    Preconditions.checkNotNull(name);

    try {
      this.packageIdentifier = packageIdentifier;
      this.name = canonicalizeTargetName(name);
    } catch (SyntaxException e) {
      // This check is just for a more helpful error message
      // i.e. valid target name, invalid package name, colon-free label form
      // used => probably they meant "//foo:bar.c" not "//foo/bar.c".
      if (packageIdentifier.getPackageFragment().getPathString().endsWith("/" + name)) {
        throw new SyntaxException(e.getMessage() + " (perhaps you meant \":" + name + "\"?)");
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
   * Returns the path fragment of the package in which this rule was declared (e.g. {@code
   * //file/base:fileutils_test} returns {@code file/base}).
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
   * @throws SyntaxException if {@code targetName} is not a valid target name
   */
  public Label getLocalTargetLabel(String targetName) throws SyntaxException {
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
      + "For example:<br><ul>"
      + "<li><code>:quux</code> relative to <code>//foo/bar:baz</code> is "
      + "<code>//foo/bar:quux</code></li>"
      + "<li><code>//wiz:quux</code> relative to <code>//foo/bar:baz</code> is "
      + "<code>//wiz:quux</code></li></ul>")
  public Label getRelative(String relName) throws SyntaxException {
    if (relName.length() == 0) {
      throw new SyntaxException("empty package-relative label");
    }
    if (relName.startsWith("//")) {
      return parseAbsolute(relName);
    } else if (relName.equals(":")) {
      throw new SyntaxException("':' is not a valid package-relative label");
    } else if (relName.charAt(0) == ':') {
      return getLocalTargetLabel(relName.substring(1));
    } else {
      return getLocalTargetLabel(relName);
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
}
