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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import java.util.Collection;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Static utility methods pertaining to restricting Starlark method invocations */
// TODO(bazel-team): Maybe we can merge this utility class with some other existing allowlist
// helper? But it seems like a lot of existing allowlist machinery is geared toward allowlists on
// rule attributes rather than what .bzl you're in.
public final class BuiltinRestriction {

  private static final String MAIN_REPO_NAME = RepositoryName.MAIN.getName();

  /** An allowlist of packages that can access restricted APIs. */
  public static final class Allowlist {
    private static final Allowlist EMPTY = new Allowlist(ImmutableList.of(), ImmutableList.of());

    // Keep separate lists for main and external repo entries. This allows us to optimize checks
    // based on the incoming label's repository.
    private final ImmutableList<AllowlistEntry> mainRepoEntries;
    private final ImmutableList<AllowlistEntry> externalRepoEntries;

    private Allowlist(
        ImmutableList<AllowlistEntry> mainRepoEntries,
        ImmutableList<AllowlistEntry> externalRepoEntries) {
      this.mainRepoEntries = mainRepoEntries;
      this.externalRepoEntries = externalRepoEntries;
    }

    public static Allowlist create(Collection<AllowlistEntry> entries) {
      if (entries.isEmpty()) {
        return EMPTY;
      }
      ImmutableList.Builder<AllowlistEntry> mainBuilder = ImmutableList.builder();
      ImmutableList.Builder<AllowlistEntry> externalBuilder = ImmutableList.builder();
      for (AllowlistEntry entry : entries) {
        if (entry.apparentRepoName().equals(MAIN_REPO_NAME)) {
          mainBuilder.add(entry);
        } else {
          externalBuilder.add(entry);
        }
      }
      return new Allowlist(mainBuilder.build(), externalBuilder.build());
    }

    public static Allowlist of(AllowlistEntry... entries) {
      return create(Arrays.asList(entries));
    }

    private boolean allows(Label label, RepositoryMapping repoMapping) {
      // Check main repo entries first to reduce the chances of needing to look up repo mappings.
      if (label.getRepository().isMain() && anyAllows(mainRepoEntries, label, repoMapping)) {
        return true;
      }
      return anyAllows(externalRepoEntries, label, repoMapping);
    }

    private static boolean anyAllows(
        ImmutableList<AllowlistEntry> entries, Label label, RepositoryMapping repoMapping) {
      // Hot code path, avoid iterator garbage.
      for (int i = 0; i < entries.size(); i++) {
        if (entries.get(i).allows(label, repoMapping)) {
          return true;
        }
      }
      return false;
    }
  }

  /**
   * The "default" allowlist for restricted APIs added to aid the Java to Starlark migration.
   *
   * <p>Entries are roughly ordered by expected call frequency (most frequent first), since {@link
   * Allowlist} checks them in order and stops at the first match.
   */
  public static final Allowlist INTERNAL_STARLARK_API_ALLOWLIST =
      Allowlist.of(
          // Cc rules
          mainRepoAllowlistEntry("third_party/bazel_rules/rules_cc"),
          mainRepoAllowlistEntry("tools/build_defs/cc"),
          externalRepoAllowlistEntry("rules_cc", ""),

          // Java rules
          mainRepoAllowlistEntry("third_party/bazel_rules/rules_java/java"),
          externalRepoAllowlistEntry("rules_java", "java"),

          // Proto rules
          mainRepoAllowlistEntry("third_party/protobuf"),
          externalRepoAllowlistEntry("protobuf", ""),
          externalRepoAllowlistEntry("com_google_protobuf", ""),

          // Rust rules
          mainRepoAllowlistEntry("devtools/rust/toolchain/testing"),
          mainRepoAllowlistEntry("third_party/bazel_rules/rules_rust/rust"),
          mainRepoAllowlistEntry("third_party/crubit"),
          externalRepoAllowlistEntry("rules_rust", "rust/private"),

          // Go rules
          mainRepoAllowlistEntry("tools/build_defs/go"),

          // BuildInfo
          mainRepoAllowlistEntry("tools/build_defs/build_info"),
          externalRepoAllowlistEntry("bazel_tools", "tools/build_defs/build_info"),

          // Android rules
          mainRepoAllowlistEntry("bazel_internal/test_rules/cc"),
          mainRepoAllowlistEntry("tools/build_defs/android"),
          mainRepoAllowlistEntry("third_party/bazel_rules/rules_android"),
          externalRepoAllowlistEntry("rules_android", ""),
          externalRepoAllowlistEntry("build_bazel_rules_android", ""),

          // Apple rules
          mainRepoAllowlistEntry("third_party/apple_crosstool"),
          mainRepoAllowlistEntry("third_party/cpptoolchains/portable_llvm/build_defs"),
          mainRepoAllowlistEntry("third_party/bazel_rules/rules_apple"),
          externalRepoAllowlistEntry("rules_apple", ""),
          externalRepoAllowlistEntry("build_bazel_rules_apple", ""),

          // CUDA rules
          mainRepoAllowlistEntry("third_party/gpus/cuda"),

          // Packaging rules
          mainRepoAllowlistEntry("tools/build_defs/packaging"),

          // Shell rules
          externalRepoAllowlistEntry("rules_shell", ""),

          // Testing
          mainRepoAllowlistEntry("test"),
          mainRepoAllowlistEntry("bazel_internal/test_rules"));

  private BuiltinRestriction() {}

  /** An entry in an {@link Allowlist}. */
  public record AllowlistEntry(String apparentRepoName, PathFragment packagePrefix) {

    public AllowlistEntry {
      checkNotNull(apparentRepoName);
      checkNotNull(packagePrefix);
    }

    private boolean allows(Label label, RepositoryMapping repoMapping) {
      return reposMatch(apparentRepoName, label.getRepository(), repoMapping)
          && label.getPackageFragment().startsWith(packagePrefix);
    }

    private static boolean reposMatch(
        String allowedName, RepositoryName givenName, RepositoryMapping repoMapping) {
      if (givenName.isMain()) {
        // The main repository may be one of the allowlisted rulesets, in which case we need to fall
        // back to interpreting allowedName as the apparent repo name. This is not a performance
        // concern since:
        // * In Bazel, the main repo is not expected to use private API unless it is one of the
        //   allowlisted rulesets. For these rulesets, it is acceptable to pay the cost of a failed
        //   RepositoryMapping lookup, which is expensive because it uses SpellChecker to construct
        //   error messages. The only other case in which this cost is paid is if the main repo
        //   attempts to use private APIs and subsequently fails.
        // * In Blaze, we should virtually always hit the first branch of the disjunction below,
        //   since Allowlist checks main repo entries first.
        return allowedName.equals(MAIN_REPO_NAME) || repoMapping.get(allowedName).isMain();
      }
      if (givenName.equals(RepositoryName.BAZEL_TOOLS)) {
        return allowedName.equals(RepositoryName.BAZEL_TOOLS.getName());
      }
      // allowedName is a module name and givenName is a real canonical repo name, so it belongs to
      // any version of that module if and only if it contains <allowedName>+ as a prefix.
      return givenName.getName().startsWith(allowedName + "+");
    }
  }

  /** Creates an {@link AllowlistEntry} in the main repository. */
  public static AllowlistEntry mainRepoAllowlistEntry(String packagePrefix) {
    return new AllowlistEntry(MAIN_REPO_NAME, PathFragment.create(packagePrefix));
  }

  /**
   * Creates an {@link AllowlistEntry} for an external repository. This is essentially an unresolved
   * package identifier; that is, a package identifier that has an apparent repo name in place of a
   * canonical repo name.
   */
  public static AllowlistEntry externalRepoAllowlistEntry(
      String apparentRepoName, String packagePrefix) {
    checkArgument(!apparentRepoName.equals(MAIN_REPO_NAME));
    return new AllowlistEntry(apparentRepoName, PathFragment.create(packagePrefix));
  }

  /**
   * Throws {@code EvalException} if the innermost Starlark function in the given thread's call
   * stack is not defined within the builtins repository.
   */
  public static void failIfCalledOutsideBuiltins(StarlarkThread thread) throws EvalException {
    failIfCalledOutsideAllowlist(thread, Allowlist.EMPTY);
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
    failIfLabelOutsideAllowlist(moduleContext.label(), moduleContext.repoMapping(), allowlist);
  }

  /**
   * Throws {@code EvalException} if the given {@link Label} is not within either 1) the builtins
   * repository, or 2) a package or subpackage of an entry in the given allowlist.
   */
  public static void failIfLabelOutsideAllowlist(
      Label label, RepositoryMapping repoMapping, Allowlist allowlist) throws EvalException {
    if (isNotAllowed(label, repoMapping, allowlist)) {
      throw Starlark.errorf("file '%s' cannot use private API", label.getCanonicalForm());
    }
  }

  /**
   * Returns true if the given {@link Label} is not within both 1) the builtins repository, or 2) a
   * package or subpackage of an entry in the given allowlist.
   */
  public static boolean isNotAllowed(
      Label label, RepositoryMapping repoMapping, Allowlist allowlist) {
    if (label.getRepository().equals(RepositoryName.BUILTINS)) {
      return false;
    }
    return !allowlist.allows(label, repoMapping);
  }
}
