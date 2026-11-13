// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.hash;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link TransitiveHashes} on real packages, as computed by {@code query}. */
@RunWith(JUnit4.class)
public final class TransitiveHashesIntegrationTest extends BuildViewTestCase {

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new AnalysisMock.Delegate(super.getAnalysisMock()) {
      @Override
      public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
          BlazeDirectories directories) {
        return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
            .putAll(super.getSkyFunctions(directories))
            .put(DirectTargetHashFunction.NAME, new DirectTargetHashFunction(directories))
            .put(SourceFileDigestFunction.NAME, new SourceFileDigestFunction())
            .put(RepoContentDigestFunction.NAME, new RepoContentDigestFunction(directories))
            .put(RegisteredToolchainsByTypeFunction.NAME, new RegisteredToolchainsByTypeFunction())
            .buildOrThrow();
      }
    };
  }

  @Before
  public void writeWorkspace() throws Exception {
    scratch.file(
        "defs/defs.bzl",
        """
        def _tc_impl(ctx):
            return [platform_common.ToolchainInfo(src = ctx.file.src)]

        my_toolchain = rule(_tc_impl, attrs = {"src": attr.label(allow_single_file = True)})

        def _impl(ctx):
            pass

        uses_toolchain = rule(
            implementation = _impl,
            attrs = {"srcs": attr.label_list(allow_files = True)},
            toolchains = ["//tc:type"],
        )

        uses_exec_group_toolchain = rule(
            implementation = _impl,
            exec_groups = {"tool": exec_group(toolchains = ["//tc:type"])},
        )

        def _aspect_impl(target, ctx):
            return []

        aspect_with_toolchain = aspect(
            implementation = _aspect_impl,
            toolchains = ["//tc:aspect_type"],
        )

        uses_aspect = rule(
            implementation = _impl,
            attrs = {"deps": attr.label_list(aspects = [aspect_with_toolchain])},
        )
        """);
    scratch.file("defs/BUILD");
    scratch.file(
        "tc/BUILD",
        """
        load("//defs:defs.bzl", "my_toolchain")

        toolchain_type(name = "type")

        toolchain_type(name = "aspect_type")

        my_toolchain(name = "impl_c", src = "c.txt")

        toolchain(
            name = "tc_c",
            toolchain = ":impl_c",
            toolchain_type = ":aspect_type",
        )

        my_toolchain(name = "impl_a", src = "a.txt")

        my_toolchain(name = "impl_b", src = "b.txt")

        toolchain(
            name = "tc_a",
            toolchain = ":impl_a",
            toolchain_type = ":type",
        )

        toolchain(
            name = "tc_b",
            toolchain = ":impl_b",
            toolchain_type = ":type",
        )

        toolchain(
            name = "tc_ext",
            toolchain = "@ext//:impl",
            toolchain_type = ":type",
        )
        """);
    scratch.file("tc/a.txt", "a");
    scratch.file("tc/b.txt", "b");
    scratch.file("tc/c.txt", "c");
    scratch.file("tc/patch.txt", "patch");
    scratch.file("tc/template.txt", "template");
    scratch.file("tc/linked.txt", "linked");
    scratch.file(
        "app/BUILD",
        """
        load("//defs:defs.bzl", "uses_aspect", "uses_exec_group_toolchain", "uses_toolchain")

        uses_toolchain(
            name = "lib",
            srcs = ["lib.txt"],
        )

        uses_exec_group_toolchain(name = "gen")

        uses_aspect(
            name = "aspect_user",
            deps = [":unrelated"],
        )

        filegroup(
            name = "unrelated",
            srcs = ["other.txt"],
        )
        """);
    scratch.file("app/lib.txt", "lib");
    scratch.file("app/other.txt", "other");
    scratch.file(
        "defs/repos.bzl",
        """
        def _impl(rctx):
            rctx.file(
                "BUILD",
                "\\n".join([
                    'load("@@//defs:defs.bzl", "my_toolchain")',
                    'filegroup(name = "files", srcs = glob(["*.txt"]), visibility = ["//visibility:public"])',
                    'my_toolchain(name = "impl", src = "a.txt", visibility = ["//visibility:public"])',
                ]),
            )
            rctx.file("a.txt", rctx.attr.content)
            rctx.file("b.txt", "b")
            rctx.watch(rctx.attr.linked)
            rctx.symlink(rctx.attr.linked, "linked.txt")

        content_repo = repository_rule(
            implementation = _impl,
            attrs = {
                "content": attr.string(),
                "unused": attr.string(),
                "patch": attr.label(),
                "template": attr.label(default = "//tc:template.txt"),
                "linked": attr.label(default = "//tc:linked.txt"),
            },
        )
        """);
    scratch.file(
        "ext_user/BUILD",
        """
        filegroup(
            name = "uses_ext",
            srcs = ["@ext//:files"],
        )
        """);
    writeModuleFile(/* content= */ "a", /* unused= */ "1");
  }

  private void writeModuleFile(String content, String unused) throws Exception {
    writeModuleFile(content, unused, "//tc:all");
  }

  private void writeModuleFile(String content, String unused, String toolchains) throws Exception {
    rewriteModuleDotBazel(
        String.format("register_toolchains('%s')", toolchains),
        "content_repo = use_repo_rule('//defs:repos.bzl', 'content_repo')",
        String.format(
            "content_repo(name = 'ext', content = '%s', unused = '%s', patch = '//tc:patch.txt')",
            content, unused));
  }

  private ImmutableMap<Label, String> extHashes() throws Exception {
    return hashes(Label.parseCanonical("//ext_user:uses_ext"));
  }

  private ImmutableMap<Label, String> hashes() throws Exception {
    return hashes(
        Label.parseCanonical("//app:lib"),
        Label.parseCanonical("//app:gen"),
        Label.parseCanonical("//app:unrelated"));
  }

  private ImmutableMap<Label, String> hashes(Label... roots) throws Exception {
    TransitiveHashes.Result result = compute(roots);
    assertThat(result.errors()).isEmpty();
    return result.hashes();
  }

  private TransitiveHashes.Result compute(Label... roots) throws Exception {
    EvaluationContext context =
        EvaluationContext.newBuilder().setParallelism(1).setEventHandler(reporter).build();
    return TransitiveHashes.compute(
        ImmutableList.copyOf(roots),
        keys -> getSkyframeExecutor().prepareAndGet(keys, context),
        message -> {});
  }

  private ImmutableMap<Label, String> hashesAfterEdit(String path, String content)
      throws Exception {
    scratch.overwriteFile(path, content);
    invalidatePackages();
    return hashes();
  }

  private static ImmutableList<String> changed(
      ImmutableMap<Label, String> before, ImmutableMap<Label, String> after) {
    return before.keySet().stream()
        .filter(label -> !before.get(label).equals(after.get(label)))
        .map(Label::toString)
        .sorted()
        .collect(ImmutableList.toImmutableList());
  }

  @Test
  public void registeredToolchain_affectsTargetsUsingItsType() throws Exception {
    ImmutableMap<Label, String> before = hashes();

    // Without a configuration, every registered toolchain counts, not just the one resolution
    // would pick.
    assertThat(changed(before, hashesAfterEdit("tc/b.txt", "changed b")))
        .containsExactly("//app:gen", "//app:lib");
  }

  @Test
  public void registeringToolchains_affectsTargetsUsingTheType() throws Exception {
    ImmutableMap<Label, String> before = hashes();

    writeModuleFile(/* content= */ "a", /* unused= */ "1", /* toolchains= */ "//tc:tc_a");

    assertThat(changed(before, hashes())).containsExactly("//app:gen", "//app:lib");
  }

  @Test
  public void externalRepoContentChange_affectsDependents() throws Exception {
    ImmutableMap<Label, String> before = extHashes();

    writeModuleFile(/* content= */ "changed", /* unused= */ "1");

    assertThat(extHashes()).isNotEqualTo(before);
  }

  @Test
  public void externalRepoRefetchedWithSameContents_doesNotAffectDependents() throws Exception {
    ImmutableMap<Label, String> before = extHashes();

    // The repo definition changes, so the repo is refetched, but its files are identical.
    writeModuleFile(/* content= */ "a", /* unused= */ "2");

    assertThat(extHashes()).isEqualTo(before);
  }

  @Test
  public void externalRepoRule_doesNotListSamePackageSourceFilesAsDeps() throws Exception {
    EvaluationContext context =
        EvaluationContext.newBuilder().setParallelism(1).setEventHandler(reporter).build();
    var userKey = DirectTargetHashFunction.key(Label.parseCanonical("//ext_user:uses_ext"));
    var user =
        (DirectTargetHashFunction.Value)
            getSkyframeExecutor().prepareAndGet(ImmutableSet.of(userKey), context).get(userKey);
    Label files = Iterables.getOnlyElement(user.getDeps());
    var filesKey = DirectTargetHashFunction.key(files);

    var result = getSkyframeExecutor().prepareAndGet(ImmutableSet.of(filesKey), context);
    assertThat(result.getError(filesKey)).isNull();
    var filesValue = (DirectTargetHashFunction.Value) result.get(filesKey);

    // a.txt and b.txt are covered by the digest of the whole repo.
    assertThat(filesValue.getDeps()).isEmpty();
  }

  @Test
  public void externalToolchainImplementation_isHashedByRepoDefinitionWithoutFetchingIt()
      throws Exception {
    ImmutableMap<Label, String> before = hashes();

    // @ext//:impl implements //tc:tc_ext. Its repo's definition counts, so a change reaches the
    // targets that use the toolchain type.
    writeModuleFile(/* content= */ "changed", /* unused= */ "1");

    assertThat(changed(before, hashes())).containsExactly("//app:gen", "//app:lib");
    EvaluationContext context =
        EvaluationContext.newBuilder().setParallelism(1).setEventHandler(reporter).build();
    var key = DirectTargetHashFunction.key(Label.parseCanonical("//tc:tc_ext"));
    var value =
        (DirectTargetHashFunction.Value)
            getSkyframeExecutor().prepareAndGet(ImmutableSet.of(key), context).get(key);
    assertThat(value.getDeps().stream().filter(dep -> !dep.getRepository().isMain())).isEmpty();
    var userKey = DirectTargetHashFunction.key(Label.parseCanonical("//ext_user:uses_ext"));
    var user =
        (DirectTargetHashFunction.Value)
            getSkyframeExecutor().prepareAndGet(ImmutableSet.of(userKey), context).get(userKey);
    var extDirectory =
        RepositoryDirectoryValue.key(Iterables.getOnlyElement(user.getDeps()).getRepository());
    assertThat(getSkyframeExecutor().getEvaluator().getExistingValue(extDirectory)).isNull();
    // A target in a package of @ext does need it fetched, which shows that the key is right.
    var unused = extHashes();
    assertThat(getSkyframeExecutor().getEvaluator().getExistingValue(extDirectory)).isNotNull();
  }

  @Test
  public void externalToolchainImplementation_isHashedByFilesItsRepoDefinitionPointsTo()
      throws Exception {
    ImmutableMap<Label, String> before = hashes();

    // The definition of @ext points to //tc:patch.txt, like a patch applied to a toolchain repo.
    assertThat(changed(before, hashesAfterEdit("tc/patch.txt", "changed patch")))
        .containsExactly("//app:gen", "//app:lib");
  }

  @Test
  public void externalToolchainImplementation_isHashedByFilesItsRepoRuleDefaultsPointTo()
      throws Exception {
    ImmutableMap<Label, String> before = hashes();

    // The default value of a label attribute of @ext's repo rule points to //tc:template.txt.
    assertThat(changed(before, hashesAfterEdit("tc/template.txt", "changed template")))
        .containsExactly("//app:gen", "//app:lib");
  }

  @Test
  public void externalRepoSymlinkIntoWorkspace_isHashedByTargetContents() throws Exception {
    ImmutableMap<Label, String> before = extHashes();

    // @ext symlinks to //tc:linked.txt with an absolute path, which stays the same when the file
    // changes.
    scratch.overwriteFile("tc/linked.txt", "changed linked");
    invalidatePackages();

    assertThat(extHashes()).isNotEqualTo(before);
  }

  @Test
  public void aspectToolchain_affectsTargetsTheAspectAppliesThrough() throws Exception {
    Label user = Label.parseCanonical("//app:aspect_user");
    ImmutableMap<Label, String> before = hashes(user);

    scratch.overwriteFile("tc/c.txt", "changed c");
    invalidatePackages();

    assertThat(hashes(user)).isNotEqualTo(before);
  }

  @Test
  public void missingDependency_isHashedWithoutError() throws Exception {
    scratch.file(
        "missing_dep/BUILD",
        """
        filegroup(
            name = "fg",
            srcs = ["//does/not:exist"],
        )
        """);

    ImmutableMap<Label, String> before = hashes(Label.parseCanonical("//missing_dep:fg"));
    scratch.file("does/not/BUILD", "filegroup(name = 'exist')");
    invalidatePackages();

    assertThat(hashes(Label.parseCanonical("//missing_dep:fg"))).isNotEqualTo(before);
  }

  @Test
  public void unavailableRegisteredToolchains_areReported() throws Exception {
    writeModuleFile(/* content= */ "a", /* unused= */ "1", /* toolchains= */ "//nonexistent:all");

    TransitiveHashes.Result result = compute(Label.parseCanonical("//app:lib"));

    assertThat(result.errors().keySet()).containsExactly(Label.parseCanonical("//tc:type"));
  }

  @Test
  public void sourceFileExecutableBit_affectsHash() throws Exception {
    ImmutableMap<Label, String> before = hashes();

    scratch.resolve("app/other.txt").setExecutable(true);
    invalidatePackages();
    ImmutableMap<Label, String> executable = hashes();
    scratch.resolve("app/other.txt").setExecutable(false);
    invalidatePackages();

    assertThat(changed(before, executable)).containsExactly("//app:unrelated");
    assertThat(hashes()).isEqualTo(before);
  }
}
