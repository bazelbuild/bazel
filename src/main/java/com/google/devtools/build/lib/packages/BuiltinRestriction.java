// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static java.util.Comparator.comparing;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Comparator;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Static utility methods pertaining to restricting Starlark method invocations */
// TODO(bazel-team): Maybe we can merge this utility class with some other existing allowlist
// helper? But it seems like a lot of existing allowlist machinery is geared toward allowlists on
// rule attributes rather than what .bzl you're in.
public final class BuiltinRestriction {

  private static final String BUILTINS_REPO_NAME = "_builtins";

  /** The "default" allowlist for restricted APIs added to aid the Java to Starlark migration. */
  public static final Allowlist INTERNAL_STARLARK_API_ALLOWLIST =
      Allowlist.of(
          ImmutableList.of(
              // Testing
              BuiltinRestriction.allowlistEntry("", "test"),
              BuiltinRestriction.allowlistEntry("", "bazel_internal/test_rules"),

              // BuildInfo
              BuiltinRestriction.allowlistEntry("", "tools/build_defs/build_info"),
              BuiltinRestriction.allowlistEntry("bazel_tools", "tools/build_defs/build_info"),

              // Android rules
              BuiltinRestriction.allowlistEntry("", "bazel_internal/test_rules/cc"),
              BuiltinRestriction.allowlistEntry("", "tools/build_defs/android"),
              BuiltinRestriction.allowlistEntry("", "third_party/bazel_rules/rules_android"),
              BuiltinRestriction.allowlistEntry("rules_android", ""),

              // Apple rules
              BuiltinRestriction.allowlistEntry("", "third_party/apple_crosstool"),
              BuiltinRestriction.allowlistEntry(
                  "", "third_party/cpptoolchains/portable_llvm/build_defs"),
              BuiltinRestriction.allowlistEntry("", "third_party/bazel_rules/rules_apple"),
              BuiltinRestriction.allowlistEntry("rules_apple", ""),

              // Cc rules
              BuiltinRestriction.allowlistEntry("", "third_party/bazel_rules/rules_cc"),
              BuiltinRestriction.allowlistEntry("", "tools/build_defs/cc"),
              BuiltinRestriction.allowlistEntry("rules_cc", ""),

              // Java rules
              BuiltinRestriction.allowlistEntry("", "third_party/bazel_rules/rules_java/java"),
              BuiltinRestriction.allowlistEntry("rules_java", "java"),

              // Rust rules
              BuiltinRestriction.allowlistEntry(
                  "", "third_party/bazel_rules/rules_rust/rust/private"),
              BuiltinRestriction.allowlistEntry("", "third_party/crubit"),
              BuiltinRestriction.allowlistEntry("rules_rust", "rust/private"),

              // CUDA rules
              BuiltinRestriction.allowlistEntry("", "third_party/gpus/cuda"),

              // Packaging rules
              BuiltinRestriction.allowlistEntry("", "tools/build_defs/packaging"),

              // Go rules
              BuiltinRestriction.allowlistEntry("", "tools/build_defs/go"),

              // Proto rules
              BuiltinRestriction.allowlistEntry("", "third_party/protobuf"),
              BuiltinRestriction.allowlistEntry("protobuf", ""),

              // Shell rules
              BuiltinRestriction.allowlistEntry("rules_shell", "")));

  private BuiltinRestriction() {}

  /**
   * Throws {@code EvalException} if the innermost Starlark function in the given thread's call
   * stack is not defined within the builtins repository.
   *
   * @throws NullPointerException if there is no currently executing Starlark function, or the
   *     innermost Starlark function's module is not a .bzl file
   */
  public static void failIfCalledOutsideBuiltins(StarlarkThread thread) throws EvalException {
    Label currentFile = BazelModuleContext.ofInnermostBzlOrThrow(thread).label();
    if (!currentFile.getRepository().getName().equals(BUILTINS_REPO_NAME)) {
      throw Starlark.errorf(
          "file '%s' cannot use private @%s API",
          currentFile.getCanonicalForm(), BUILTINS_REPO_NAME);
    }
  }

  /** An allowlist for subpackages of a list of packages and possibly entire repositories. */
  public static final class Allowlist {
    private static final Comparator<PackageIdentifier> HIERARCHICAL_COMPARATOR =
        comparing(
                (PackageIdentifier packageIdentifier) ->
                    packageIdentifier.getRepository().getName())
            .thenComparing(
                PackageIdentifier::getPackageFragment, PathFragment.HIERARCHICAL_COMPARATOR);

    public static final Allowlist BUILTINS_ONLY = new Allowlist(ImmutableSortedSet.of());

    private final ImmutableSortedSet<PackageIdentifier> allowedPrefixes;

    private Allowlist(ImmutableSortedSet<PackageIdentifier> allowedPrefixes) {
      this.allowedPrefixes = allowedPrefixes;
    }

    public static Allowlist of(Collection<AllowlistEntry> entries) {
      var builder = ImmutableSortedSet.<PackageIdentifier>orderedBy(HIERARCHICAL_COMPARATOR);
      for (var entry : entries) {
        String repoNamePrefix =
            switch (entry.apparentRepoName()) {
              // These special repos preserve their apparent name as their canonical name. Also
              // preserve the main repo as is as it's treated specially below.
              // Note: This omits the platforms repo, but it will never use builtins.
              case "", "bazel_tools", BUILTINS_REPO_NAME -> entry.apparentRepoName();
              // All other apparent module names are interpreted as module names, which map to a
              // repo of the form <name>+[<version>].
              default -> {
                try {
                  RepositoryName.validateUserProvidedRepoName(entry.apparentRepoName());
                } catch (EvalException e) {
                  // Privileged and unsupported API, so a Bazel crash is acceptable.
                  throw new IllegalArgumentException(
                      "Invalid apparent repo name in builtin allowlist: "
                          + entry.apparentRepoName(),
                      e);
                }
                yield entry.apparentRepoName() + "+";
              }
            };
        builder.add(
            PackageIdentifier.create(
                RepositoryName.createUnvalidated(repoNamePrefix), entry.packagePrefix()));
      }
      return new Allowlist(builder.build());
    }

    /**
     * Whether the given package is contained in the allowlist. The _builtins repo is contained in
     * any allowlist, even if not explicitly provided.
     */
    public boolean allows(PackageIdentifier entry) {
      if (entry.getRepository().getName().equals(BUILTINS_REPO_NAME)) {
        return true;
      }
      // If an entry in the allowlist contains prefixes for both the repo and the package name, it
      // necessarily sorts right before the given entry since both repo and package name are
      // compared in lexicographic order and no canonical repo name produced by Bazel can have two
      // different repo names in the allowlist as true prefixes.
      var floorEntry = allowedPrefixes.floor(entry);
      return floorEntry != null
          && reposMatch(floorEntry.getRepository(), entry.getRepository())
          && entry.getPackageFragment().startsWith(floorEntry.getPackageFragment());
    }

    private static boolean reposMatch(RepositoryName allowedName, RepositoryName givenName) {
      if (allowedName.isMain()) {
        return givenName.isMain();
      }
      if (allowedName.equals(RepositoryName.BAZEL_TOOLS)) {
        return givenName.equals(RepositoryName.BAZEL_TOOLS);
      }
      if (allowedName.equals(RepositoryName.BUILTINS)) {
        return givenName.equals(RepositoryName.BUILTINS);
      }
      // allowedName is of the form <module name>+ and givenName is a real canonical repo name, so
      // it belongs to any version of that module if and only if it contains allowedName as a
      // prefix.
      return givenName.getName().startsWith(allowedName.getName());
    }
  }

  /**
   * An entry in an allowlist that can be checked using {@link #failIfCalledOutsideAllowlist} or
   * {@link #failIfModuleOutsideAllowlist}.
   */
  public record AllowlistEntry(String apparentRepoName, PathFragment packagePrefix) {}

  /**
   * Creates an {@link AllowlistEntry}. This is essentially an unresolved package identifier; that
   * is, a package identifier that has an apparent repo name in place of a canonical repo name.
   */
  public static AllowlistEntry allowlistEntry(String repoNamePrefix, String packagePrefix) {
    return new AllowlistEntry(repoNamePrefix, PathFragment.create(packagePrefix));
  }

  /**
   * Throws {@code EvalException} if the innermost Starlark function in the given thread's call
   * stack is not defined within either 1) the builtins repository, or 2) a package or subpackage of
   * an entry in the given allowlist.
   *
   * @throws NullPointerException if there is no currently executing Starlark function, or the
   *     innermost Starlark function's module is not a .bzl file
   */
  public static void failIfCalledOutsideAllowlist(StarlarkThread thread, Allowlist allowlist)
      throws EvalException {
    failIfModuleOutsideAllowlist(BazelModuleContext.ofInnermostBzlOrThrow(thread), allowlist);
  }

  /**
   * Throws {@code EvalException} if the call is made outside of the default allowlist or outside of
   * builtins.
   *
   * @throws NullPointerException if there is no currently executing Starlark function, or the
   *     innermost Starlark function's module is not a .bzl file
   */
  public static void failIfCalledOutsideDefaultAllowlist(StarlarkThread thread)
      throws EvalException {
    failIfCalledOutsideAllowlist(thread, INTERNAL_STARLARK_API_ALLOWLIST);
  }

  /**
   * Throws {@code EvalException} if the given {@link BazelModuleContext} is not within either 1)
   * the builtins repository, or 2) a package or subpackage of an entry in the given allowlist.
   */
  public static void failIfModuleOutsideAllowlist(
      BazelModuleContext moduleContext, Allowlist allowlist) throws EvalException {
    failIfLabelOutsideAllowlist(moduleContext.label(), allowlist);
  }

  /**
   * Throws {@code EvalException} if the given {@link Label} is not within either 1) the builtins
   * repository, or 2) a package or subpackage of an entry in the given allowlist.
   */
  public static void failIfLabelOutsideAllowlist(Label label, Allowlist allowlist)
      throws EvalException {
    if (!allowlist.allows(label.getPackageIdentifier())) {
      throw Starlark.errorf("file '%s' cannot use private API", label.getCanonicalForm());
    }
  }
}
