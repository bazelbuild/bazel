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

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.LabelValidator.BadLabelException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.Canonicalizer;
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
public final class Label implements Comparable<Label>, Serializable, ClassObject {

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
  public static Label parseWorkspaceLabel(String absName) throws SyntaxException {
    String repo = null;
    int packageStartPos = absName.indexOf("//");
    if (packageStartPos > 0) {
      repo = absName.substring(0, packageStartPos);
      absName = absName.substring(packageStartPos);
    }
    try {
      LabelValidator.PackageAndTarget labelParts = LabelValidator.parseAbsoluteLabel(absName);
      return new Label(repo, labelParts.getPackageName(), labelParts.getTargetName());
    } catch (BadLabelException e) {
      throw new SyntaxException(e.getMessage());
    }
  }

  /**
   * Factory for Labels from absolute string form. e.g.
   * <pre>
   * //foo/bar
   * //foo/bar:quux
   * //foo/bar:      (undocumented, but accepted)
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
    return new Label(path.getPathString(), label.substring(index+1));
  }

  /**
   * Validates the given repository name and returns a canonical String instance if it is valid.
   * Otherwise throws a SyntaxException.
   * @throws SyntaxException
   */
  private static String canonicalizeWorkspaceName(String workspaceName) throws SyntaxException {
    String error = LabelValidator.validateWorkspaceName(workspaceName);
    if (error != null) {
      error = "invalid workspace name '" + StringUtilities.sanitizeControlChars(workspaceName)
          + "': " + error;
      throw new SyntaxException(error);
    }

    return StringCanonicalizer.intern(workspaceName);
  }

  /**
   * Validates the given package name and returns a canonical PathFragment instance if it is valid.
   * Otherwise it throws a SyntaxException.
   */
  private static PathFragment canonicalizePackageName(String packageName, String name)
      throws SyntaxException {
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
    return Canonicalizer.fragments().intern(new PathFragment(packageName));
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

    // TODO(bazel-team): This should be an error, but we turn out to have around ~300 instances of
    // this in the depot.
    if (name.endsWith("/.")) {
      name = name.substring(0, name.length() - 2);
    }

    return StringCanonicalizer.intern(name);
  }

  /** The name of the workspace. */
  private Optional<String> workspaceName;

  /** The name of the package. Canonical (i.e. x.equals(y) <=> x==y). */
  private PathFragment packageName;

  /** The name of the target within the package. Canonical. */
  private String name;

  /**
   * Constructor from a package name, target name. Both are checked for validity
   * and a SyntaxException is thrown if either is invalid.
   */
  private Label(String packageName, String name) throws SyntaxException {
    this(canonicalizePackageName(packageName, name), name);
  }

  /**
   * Constructor from canonical valid package name and a target name. The target
   * name is checked for validity and a SyntaxException is throw if it isn't.
   */
  private Label(PathFragment packageName, String name) throws SyntaxException {
    Preconditions.checkNotNull(packageName);
    Preconditions.checkNotNull(name);

    this.packageName = packageName;
    this.name = canonicalizeTargetName(name);
    this.workspaceName = Optional.absent();
  }

  private Label(String workspaceName, String packageName, String name) throws SyntaxException {
    this(packageName, name);
    if (workspaceName != null) {
      this.workspaceName = Optional.of(canonicalizeWorkspaceName(workspaceName));
    }
  }

  private Object writeReplace() {
    return new LabelSerializationProxy(toString());
  }

  private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Serialization is allowed only by proxy");
  }

  /**
   * Returns the name of the package in which this rule was declared (e.g. {@code
   * //file/base:fileutils_test} returns {@code file/base}).
   */
  public String getPackageName() {
    return packageName.getPathString();
  }

  /**
   * Returns the path fragment of the package in which this rule was declared (e.g. {@code
   * //file/base:fileutils_test} returns {@code file/base}).
   */
  public PathFragment getPackageFragment() {
    return packageName;
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
    return packageName.getRelative(name);
  }

  @Override
  public Object getValue(String name) {
    if (name.equals("name")) {
      return this.name;
    }
    return null;
  }

  /**
   * Returns the name by which this rule was declared (e.g. {@code //foo/bar:baz}
   * returns {@code baz}).
   */
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
    return workspaceName.or("") + "//" + packageName + ":" + name;
  }

  /**
   * Renders this label in shorthand form.
   *
   * <p>Labels with canonical form {@code //foo/bar:bar} have the shorthand form {@code //foo/bar}.
   * All other labels have identical shorthand and canonical forms.
   */
  public String toShorthandString() {
    return packageName.getBaseName().equals(name)
        ? "//" + packageName
        : toString();
  }

  /**
   * Returns a label in the same package as this label with the given target name.
   *
   * @throws SyntaxException if {@code targetName} is not a valid target name
   */
  public Label getLocalTargetLabel(String targetName) throws SyntaxException {
    return new Label(packageName, targetName);
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
    return name.hashCode() ^ packageName.hashCode();
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
        && packageName.equals(otherLabel.packageName);
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
    return packageName == other.packageName
        ? name.compareTo(other.name)
        : packageName.compareTo(other.packageName);
  }

  /**
   * Returns a suitable string for the user-friendly representation of the Label. Works even if the
   * argument is null.
   */
  public static String print(Label label) {
    return label == null ? "(unknown)" : label.toString();
  }
}
