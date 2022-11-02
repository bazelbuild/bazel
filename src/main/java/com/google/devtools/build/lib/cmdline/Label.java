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

import static com.google.devtools.build.lib.cmdline.LabelParser.validateAndProcessTargetName;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.cmdline.LabelParser.Parts;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Arrays;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
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
public final class Label implements Comparable<Label>, StarlarkValue, SkyKey, CommandLineItem {
  /**
   * Package names that aren't made relative to the current repository because they mean special
   * things to Bazel.
   */
  private static final ImmutableSet<String> ABSOLUTE_PACKAGE_NAMES =
      ImmutableSet.of(
          // Used for select's `//conditions:default` label (not a target)
          "conditions",
          // Used for the public and private visibility labels (not targets)
          "visibility",
          // There is only one //external package
          LabelConstants.EXTERNAL_PACKAGE_NAME.getPathString());

  // Intern "__pkg__" and "__subpackages__" pseudo-targets, which appears in labels used for
  // visibility specifications. This saves a couple tenths of a percent of RAM off the loading
  // phase. Note that general interning of all values for `name` is *not* beneficial. See
  // Google-internal cl/386077913 and cl/185394812 for more context.
  private static final String PKG_VISIBILITY_NAME = "__pkg__";
  private static final String SUBPACKAGES_VISIBILITY_NAME = "__subpackages__";

  public static final SkyFunctionName TRANSITIVE_TRAVERSAL =
      SkyFunctionName.createHermetic("TRANSITIVE_TRAVERSAL");

  private static final Interner<Label> LABEL_INTERNER = BlazeInterners.newWeakInterner();

  /** The context of a current repo, necessary to parse a repo-relative label ("//foo:bar"). */
  public interface RepoContext {
    static RepoContext of(RepositoryName currentRepo, RepositoryMapping repoMapping) {
      return new AutoValue_Label_RepoContextImpl(currentRepo, repoMapping);
    }

    RepositoryName currentRepo();

    RepositoryMapping repoMapping();
  }

  @AutoValue
  abstract static class RepoContextImpl implements RepoContext {}

  /** The context of a current package, necessary to parse a package-relative label (":foo"). */
  public interface PackageContext extends RepoContext {
    static PackageContext of(PackageIdentifier currentPackage, RepositoryMapping repoMapping) {
      return new AutoValue_Label_PackageContextImpl(
          currentPackage.getRepository(), repoMapping, currentPackage.getPackageFragment());
    }

    PathFragment packageFragment();

    default PackageIdentifier packageIdentifier() {
      return PackageIdentifier.create(currentRepo(), packageFragment());
    }
  }

  @AutoValue
  abstract static class PackageContextImpl implements PackageContext {}

  /**
   * Parses a raw label string that contains the canonical form of a label. It must be of the form
   * {@code [@repo]//foo/bar[:quux]}. If the {@code @repo} part is present, it must be a canonical
   * repo name, otherwise the label will be assumed to be in the main repo.
   */
  public static Label parseCanonical(String raw) throws LabelSyntaxException {
    Parts parts = Parts.parse(raw);
    parts.checkPkgIsAbsolute();
    RepositoryName repoName =
        parts.repo == null ? RepositoryName.MAIN : RepositoryName.createUnvalidated(parts.repo);
    return createUnvalidated(
        PackageIdentifier.create(repoName, PathFragment.create(parts.pkg)), parts.target);
  }

  public static Label parseCanonicalUnchecked(String raw) {
    try {
      return parseCanonical(raw);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(e);
    }
  }

  /** Computes the repo name for the label, within the context of a current repo. */
  private static RepositoryName computeRepoNameWithRepoContext(
      Parts parts, RepoContext repoContext) {
    if (parts.repo == null) {
      // Certain package names when used without a "@" part are always absolutely in the main repo,
      // disregarding the current repo and repo mappings.
      return ABSOLUTE_PACKAGE_NAMES.contains(parts.pkg)
          ? RepositoryName.MAIN
          : repoContext.currentRepo();
    }
    if (parts.repoIsCanonical) {
      // This label uses the canonical label literal syntax starting with two @'s ("@@foo//bar").
      return RepositoryName.createUnvalidated(parts.repo);
    }
    return repoContext.repoMapping().get(parts.repo);
  }

  /**
   * Parses a raw label string within the context of a current repo. It must be of the form {@code
   * [@repo]//foo/bar[:quux]}. If the {@code @repo} part is present, it will undergo {@code
   * repoContext.repoMapping()}, otherwise the label will be assumed to be in {@code
   * repoContext.currentRepo()}.
   */
  public static Label parseWithRepoContext(String raw, RepoContext repoContext)
      throws LabelSyntaxException {
    Parts parts = Parts.parse(raw);
    parts.checkPkgIsAbsolute();
    RepositoryName repoName = computeRepoNameWithRepoContext(parts, repoContext);
    return createUnvalidated(
        PackageIdentifier.create(repoName, PathFragment.create(parts.pkg)), parts.target);
  }

  /**
   * Parses a raw label string within the context of a current package. It can be of a
   * package-relative form ({@code :quux}). Otherwise, it must be of the form {@code
   * [@repo]//foo/bar[:quux]}. If the {@code @repo} part is present, it will undergo {@code
   * packageContext.repoMapping()}, otherwise the label will be assumed to be in the repo of {@code
   * packageContext.currentRepo()}.
   */
  public static Label parseWithPackageContext(String raw, PackageContext packageContext)
      throws LabelSyntaxException {
    Parts parts = Parts.parse(raw);
    // pkg is either absolute or empty
    if (!parts.pkg.isEmpty()) {
      parts.checkPkgIsAbsolute();
    }
    RepositoryName repoName = computeRepoNameWithRepoContext(parts, packageContext);
    PathFragment pkgFragment =
        parts.pkgIsAbsolute ? PathFragment.create(parts.pkg) : packageContext.packageFragment();
    return createUnvalidated(PackageIdentifier.create(repoName, pkgFragment), parts.target);
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
   * <p>Labels that don't begin with a repository name are considered to be in the main repository,
   * so for instance {@code //foo/bar} will turn into {@code @//foo/bar}.
   *
   * <p>Labels that begin with a repository name will undergo {@code repositoryMapping}.
   *
   * @param absName label-like string to be parsed
   * @param repositoryMapping map of repository names from the local name found in the current
   *     repository to the global name declared in the main repository
   */
  // TODO(b/200024947): Remove this.
  public static Label parseAbsolute(String absName, RepositoryMapping repositoryMapping)
      throws LabelSyntaxException {
    Preconditions.checkNotNull(repositoryMapping);
    return parseWithRepoContext(absName, RepoContext.of(RepositoryName.MAIN, repositoryMapping));
  }

  // TODO(b/200024947): Remove this.
  public static Label parseAbsolute(
      String absName, ImmutableMap<String, RepositoryName> repositoryMapping)
      throws LabelSyntaxException {
    return parseAbsolute(absName, RepositoryMapping.createAllowingFallback(repositoryMapping));
  }

  /**
   * Alternate factory method for Labels from absolute strings. This is a convenience method for
   * cases when a Label needs to be initialized statically, so the declared exception is
   * inconvenient.
   *
   * <p>Do not use this when the argument is not hard-wired.
   */
  // TODO(b/200024947): Remove this.
  public static Label parseAbsoluteUnchecked(String absName) {
    try {
      return parseCanonical(absName);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(e);
    }
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
  // TODO(b/200024947): Remove this...?
  public static Label create(String packageName, String targetName) throws LabelSyntaxException {
    return createUnvalidated(
        PackageIdentifier.parse(packageName),
        validateAndProcessTargetName(packageName, targetName));
  }

  /**
   * Similar factory to above, but takes a package identifier to allow external repository labels to
   * be created.
   */
  // TODO(b/200024947): Remove this...?
  public static Label create(PackageIdentifier packageId, String targetName)
      throws LabelSyntaxException {
    return createUnvalidated(
        packageId,
        validateAndProcessTargetName(packageId.getPackageFragment().getPathString(), targetName));
  }

  /**
   * Similar factory to above, but does not perform target name validation.
   *
   * <p>Only call this method if you know what you're doing; in particular, don't call it on
   * arbitrary {@code name} inputs
   */
  @AutoCodec.Instantiator
  public static Label createUnvalidated(PackageIdentifier packageIdentifier, String name) {
    String internedName = name;
    if (internedName.equals(PKG_VISIBILITY_NAME)) {
      internedName = PKG_VISIBILITY_NAME;
    } else if (internedName.equals(SUBPACKAGES_VISIBILITY_NAME)) {
      internedName = SUBPACKAGES_VISIBILITY_NAME;
    }
    return LABEL_INTERNER.intern(new Label(packageIdentifier, internedName));
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
  public String getWorkspaceRootForStarlarkOnly(StarlarkSemantics semantics) throws EvalException {
    checkRepoVisibilityForStarlark("workspace_root");
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
   *
   * <p>The package's repository is not included in the returned fragment. To account for it,
   * compose this with {@code #getRepository()#getExecPath}.
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
   * <p>invariant: {@code parseCanonical(x.toString()).equals(x)}. Note that using {@link
   * #parseWithPackageContext} or {@link #parseWithRepoContext} on the returned string might not
   * yield the same label! For that, use {@link #getUnambiguousCanonicalForm()}.
   */
  @Override
  public String toString() {
    return getCanonicalForm();
  }

  /**
   * Renders this label in canonical form.
   *
   * <p>invariant: {@code parseCanonical(x.getCanonicalForm()).equals(x)}. Note that using {@link
   * #parseWithPackageContext} or {@link #parseWithRepoContext} on the returned string might not
   * yield the same label! For that, use {@link #getUnambiguousCanonicalForm()}.
   */
  public String getCanonicalForm() {
    return packageIdentifier.getCanonicalForm() + ":" + name;
  }

  /**
   * Returns an absolutely unambiguous canonical form for this label. Parsing this string in any
   * environment should yield the same label (as in {@code
   * Label.parse*(x.getUnambiguousCanonicalForm(), ...).equals(x)}).
   */
  public String getUnambiguousCanonicalForm() {
    return packageIdentifier.getUnambiguousCanonicalForm() + ":" + name;
  }

  /**
   * Returns a label string that is suitable for display, i.e., it resolves to this label when
   * parsed in the context of the main repository and has a repository part that is as simple as
   * possible.
   *
   * @param mainRepositoryMapping the {@link RepositoryMapping} of the main repository
   * @return analogous to {@link PackageIdentifier#getDisplayForm(RepositoryMapping)}
   */
  public String getDisplayForm(RepositoryMapping mainRepositoryMapping) {
    return packageIdentifier.getDisplayForm(mainRepositoryMapping) + ":" + name;
  }

  /** Return the name of the repository label refers to without the leading `at` symbol. */
  @StarlarkMethod(
      name = "workspace_name",
      structField = true,
      doc =
          "The repository part of this label. For instance, "
              + "<pre class=language-python>Label(\"@foo//bar:baz\").workspace_name"
              + " == \"foo\"</pre>")
  public String getWorkspaceName() throws EvalException {
    checkRepoVisibilityForStarlark("workspace_name");
    return packageIdentifier.getRepository().getName();
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
      repository = packageIdentifier.getRepository().getNameWithAt();
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
   * @param thread the Starlark thread.
   */
  @StarlarkMethod(
      name = "relative",
      doc =
          // TODO(#14503): Fix the documentation.
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
    return getRelativeWithRemapping(
        relName,
        BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(thread)).repoMapping());
  }

  /**
   * Resolves a relative or absolute label name. If given name is absolute, then this method calls
   * {@link #parseAbsolute}. Otherwise, it calls {@link #getLocalTargetLabel}.
   *
   * <p>For example: {@code :quux} relative to {@code //foo/bar:baz} is {@code //foo/bar:quux};
   * {@code //wiz:quux} relative to {@code //foo/bar:baz} is {@code //wiz:quux};
   * {@code @repo//foo:bar} relative to anything will be {@code @repo//foo:bar} if {@code @repo} is
   * not in {@code repositoryMapping} but will be {@code @other_repo//foo:bar} if there is an entry
   * {@code @repo -> @other_repo} in {@code repositoryMapping}.
   *
   * @param relName the relative label name; must be non-empty
   * @param repositoryMapping the map of local repository names in external repository to global
   *     repository names in main repo; can be empty, but not null
   */
  // TODO(b/200024947): Remove this.
  public Label getRelativeWithRemapping(String relName, RepositoryMapping repositoryMapping)
      throws LabelSyntaxException {
    Preconditions.checkNotNull(repositoryMapping);
    if (relName.isEmpty()) {
      throw new LabelSyntaxException("empty package-relative label");
    }
    return parseWithPackageContext(
        relName, PackageContext.of(packageIdentifier, repositoryMapping));
  }

  // TODO(b/200024947): Remove this.
  public Label getRelativeWithRemapping(
      String relName, ImmutableMap<String, RepositoryName> repositoryMapping)
      throws LabelSyntaxException {
    return getRelativeWithRemapping(
        relName, RepositoryMapping.createAllowingFallback(repositoryMapping));
  }

  @Override
  public SkyFunctionName functionName() {
    return TRANSITIVE_TRAVERSAL;
  }

  @Override
  public int hashCode() {
    return hashCode(name, packageIdentifier);
  }

  /**
   * Specialization of {@link Arrays#hashCode()} that does not require constructing a 2-element
   * array.
   */
  private static int hashCode(Object obj1, Object obj2) {
    int result = 31 + (obj1 == null ? 0 : obj1.hashCode());
    return 31 * result + (obj2 == null ? 0 : obj2.hashCode());
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
    // Package identifiers are (weakly) interned so we compare them first.
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
    String name = label.name;
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
    // TODO(wyv): Consider using StarlarkSemantics here too for optional unambiguity.
    printer.append("Label(");
    printer.repr(getCanonicalForm());
    printer.append(")");
  }

  @Override
  public void str(Printer printer, StarlarkSemantics semantics) {
    if (getRepository().isMain()
        && !semantics.getBool(
            BuildLanguageOptions.INCOMPATIBLE_UNAMBIGUOUS_LABEL_STRINGIFICATION)) {
      // If this label is in the main repo and we're not using unambiguous label stringification,
      // the result should always be "//foo:bar".
      printer.append(getCanonicalForm());
      return;
    }

    if (semantics.getBool(BuildLanguageOptions.ENABLE_BZLMOD)) {
      // If Bzlmod is enabled, we use canonical label literal syntax here and prepend an extra '@'.
      // So the result looks like "@@//foo:bar" for the main repo and "@@foo~1.0//bar:quux" for
      // other repos.
      printer.append(getUnambiguousCanonicalForm());
      return;
    }
    // If Bzlmod is not enabled, we just use a single '@'.
    // So the result looks like "@//foo:bar" for the main repo and "@foo//bar:quux" for other repos.
    printer.append(
        String.format(
            "@%s//%s:%s",
            packageIdentifier.getRepository().getName(),
            packageIdentifier.getPackageFragment(),
            name));
  }

  @Override
  public String expandToCommandLine() {
    // TODO(wyv): Consider using StarlarkSemantics here too for optional unambiguity.
    return getCanonicalForm();
  }

  private void checkRepoVisibilityForStarlark(String method) throws EvalException {
    if (!getRepository().isVisible()) {
      throw Starlark.errorf("'%s' is not allowed on invalid Label %s", method, this);
    }
  }
}
