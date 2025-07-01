// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.starlark.StarlarkGlobalsImpl;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TargetRecorder.MacroFrame;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Optional;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.SyntaxError;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PackagePiece}. */
// TODO(https://github.com/bazelbuild/bazel/issues/23852): add tests that really evaluate Starlark
// (requires package piece support in PackageManager); test getTarget error case,
// tryGetTargetRecursingUp, checkMacroNamespaceCompliance, etc.
@RunWith(JUnit4.class)
public final class PackagePieceTest {

  private static final RuleClass FAUX_TEST_CLASS =
      new RuleClass.Builder("faux_test", RuleClassType.TEST, /* starlark= */ false)
          .addAttribute(
              Attribute.attr("tags", Types.STRING_LIST).nonconfigurable("tags aren't").build())
          .addAttribute(Attribute.attr("size", Type.STRING).nonconfigurable("size isn't").build())
          .addAttribute(Attribute.attr("timeout", Type.STRING).build())
          .addAttribute(Attribute.attr("flaky", Type.BOOLEAN).build())
          .addAttribute(Attribute.attr("shard_count", Type.INTEGER).build())
          .addAttribute(Attribute.attr("local", Type.BOOLEAN).build())
          .setConfiguredTargetFunction(mock(StarlarkCallable.class))
          .build();

  private static final Label FAKE_BZL_LABEL = Label.parseCanonicalUnchecked("//fake:fake.bzl");

  private FileSystem fileSystem;
  private StarlarkFunction noopMacroImplementation;
  private StarlarkFunction failMacroImplementation;

  @Before
  public void setUp() throws SyntaxError.Exception, EvalException, InterruptedException {
    this.fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);

    ParserInput input =
        ParserInput.fromLines(
"""
def noop_impl(name, visibility, **kwargs):
    pass

def fail_impl(name, visibility, **kwargs):
    fail("always fails")
""");

    Module module =
        Module.withPredeclared(
            StarlarkSemantics.DEFAULT, StarlarkGlobalsImpl.INSTANCE.getUtilToplevels());
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      Starlark.execFile(input, FileOptions.DEFAULT, module, thread);
      this.noopMacroImplementation = (StarlarkFunction) module.getGlobal("noop_impl");
      this.failMacroImplementation = (StarlarkFunction) module.getGlobal("fail_impl");
    }
  }

  @Test
  public void identifier_equality() throws Exception {

    new EqualsTester()
        .addEqualityGroup(
            new PackagePieceIdentifier.ForBuildFile(PackageIdentifier.createInMainRepo("test_pkg")),
            new PackagePieceIdentifier.ForBuildFile(PackageIdentifier.createInMainRepo("test_pkg")))
        .addEqualityGroup(
            new PackagePieceIdentifier.ForBuildFile(PackageIdentifier.parse("@repo//test_pkg")))
        .addEqualityGroup(
            new PackagePieceIdentifier.ForMacro(
                PackageIdentifier.createInMainRepo("test_pkg"),
                new PackagePieceIdentifier.ForBuildFile(
                    PackageIdentifier.createInMainRepo("test_pkg")),
                "foo"),
            new PackagePieceIdentifier.ForMacro(
                PackageIdentifier.createInMainRepo("test_pkg"),
                new PackagePieceIdentifier.ForBuildFile(
                    PackageIdentifier.createInMainRepo("test_pkg")),
                "foo"))
        .addEqualityGroup(
            new PackagePieceIdentifier.ForMacro(
                PackageIdentifier.createInMainRepo("test_pkg"),
                new PackagePieceIdentifier.ForMacro(
                    PackageIdentifier.createInMainRepo("test_pkg"),
                    new PackagePieceIdentifier.ForBuildFile(
                        PackageIdentifier.createInMainRepo("test_pkg")),
                    "foo"),
                "foo_bar"),
            new PackagePieceIdentifier.ForMacro(
                PackageIdentifier.createInMainRepo("test_pkg"),
                new PackagePieceIdentifier.ForMacro(
                    PackageIdentifier.createInMainRepo("test_pkg"),
                    new PackagePieceIdentifier.ForBuildFile(
                        PackageIdentifier.createInMainRepo("test_pkg")),
                    "foo"),
                "foo_bar"))
        .testEquals();
  }

  @Test
  public void packagePieceForBuildFileBuilder_basicFunctionality() throws Exception {
    PackagePiece.ForBuildFile.Builder builder = minimalBuildFilePieceBuilder("test_pkg");
    addRule(builder, Label.parseCanonical("//test_pkg:foo"), FAUX_TEST_CLASS);
    MacroClass macroClass = failMacroClass("my_macro"); // would fail if expanded
    addMacro(builder, macroClass, "bar");
    PackagePiece.ForBuildFile buildFilePiece = builder.buildPartial().finishBuild();
    assertThat(buildFilePiece.getPackageIdentifier())
        .isEqualTo(PackageIdentifier.createInMainRepo("test_pkg"));
    assertThat(buildFilePiece.getMetadata().buildFileLabel())
        .isEqualTo(Label.parseCanonical("//test_pkg:BUILD"));
    assertThat(buildFilePiece.getBuildFile().getLabel())
        .isEqualTo(Label.parseCanonical("//test_pkg:BUILD"));
    assertThat(buildFilePiece.getTargets()).hasSize(2); // BUILD file + foo
    assertThat(buildFilePiece.getTargets(Rule.class)).hasSize(1);
    Target foo = buildFilePiece.getTarget("foo");
    assertThat(foo).isNotNull();
    assertThat(foo.getLabel()).isEqualTo(Label.parseCanonical("//test_pkg:foo"));
    assertThat(foo.getRuleClass()).isEqualTo(FAUX_TEST_CLASS.getName());
    assertThat(foo.getPackageoid()).isSameInstanceAs(buildFilePiece);
    assertThat(foo.getDeclaringMacro()).isNull();
    assertThat(foo.getDeclaringPackage()).isEqualTo(PackageIdentifier.createInMainRepo("test_pkg"));
    MacroInstance bar = buildFilePiece.getMacroByName("bar");
    assertThat(bar).isNotNull();
    assertThat(bar.getName()).isEqualTo("bar");
    assertThat(bar.getMacroClass()).isSameInstanceAs(macroClass);
    assertThat(bar.getPackageMetadata()).isSameInstanceAs(buildFilePiece.getMetadata());
  }

  @Test
  public void packagePieceForMacroBuilder_basicFunctionality() throws Exception {
    MacroClass fooMacroClass = noopMacroClass("foo_macro");
    MacroClass barMacroClass = noopMacroClass("bar_macro");
    PackagePiece.ForBuildFile.Builder buildFilePieceBuilder =
        minimalBuildFilePieceBuilder("test_pkg");
    MacroInstance fooMacro = addMacro(buildFilePieceBuilder, fooMacroClass, "foo");
    PackagePiece.ForBuildFile buildFilePiece = buildFilePieceBuilder.buildPartial().finishBuild();
    PackagePiece.ForMacro.Builder fooMacroPieceBuilder =
        minimalMacroPieceBuilder(fooMacro, buildFilePiece.getIdentifier(), buildFilePiece);
    // Normally, the macro frame would be set by MacroClass#executeMacroImplementation
    var unused = fooMacroPieceBuilder.setCurrentMacroFrame(new MacroFrame(fooMacro));
    addRule(fooMacroPieceBuilder, Label.parseCanonical("//test_pkg:foo_test"), FAUX_TEST_CLASS);
    MacroInstance fooBarMacro = addMacro(fooMacroPieceBuilder, barMacroClass, "foo_bar");
    PackagePiece.ForMacro fooMacroPiece = fooMacroPieceBuilder.buildPartial().finishBuild();
    PackagePiece.ForMacro.Builder fooBarMacroPieceBuilder =
        minimalMacroPieceBuilder(fooBarMacro, fooMacroPiece.getIdentifier(), buildFilePiece);
    unused = fooBarMacroPieceBuilder.setCurrentMacroFrame(new MacroFrame(fooBarMacro));
    addRule(
        fooBarMacroPieceBuilder, Label.parseCanonical("//test_pkg:foo_bar_test"), FAUX_TEST_CLASS);
    PackagePiece.ForMacro fooBarMacroPiece = fooBarMacroPieceBuilder.buildPartial().finishBuild();

    assertThat(fooMacroPiece.getEvaluatedMacro()).isSameInstanceAs(fooMacro);
    assertThat(fooBarMacroPiece.getEvaluatedMacro()).isSameInstanceAs(fooBarMacro);

    assertThat(fooMacroPiece.getMetadata()).isSameInstanceAs(buildFilePiece.getMetadata());
    assertThat(fooMacroPiece.getDeclarations()).isSameInstanceAs(buildFilePiece.getDeclarations());
    assertThat(fooBarMacroPiece.getMetadata()).isSameInstanceAs(buildFilePiece.getMetadata());
    assertThat(fooBarMacroPiece.getDeclarations())
        .isSameInstanceAs(buildFilePiece.getDeclarations());

    assertThat(fooMacroPiece.getTargets()).hasSize(1);
    assertThat(fooMacroPiece.getTargets(Rule.class)).hasSize(1);
    Target fooTest = fooMacroPiece.getTarget("foo_test");
    assertThat(fooTest).isNotNull();
    assertThat(fooTest.getLabel()).isEqualTo(Label.parseCanonical("//test_pkg:foo_test"));
    assertThat(fooTest.getRuleClass()).isEqualTo(FAUX_TEST_CLASS.getName());
    assertThat(fooTest.getPackageoid()).isSameInstanceAs(fooMacroPiece);
    assertThat(fooTest.getDeclaringMacro()).isSameInstanceAs(fooMacro);
    assertThat(fooTest.getDeclaringPackage()).isEqualTo(FAKE_BZL_LABEL.getPackageIdentifier());

    assertThat(fooMacroPiece.getMacroByName("foo_bar")).isSameInstanceAs(fooBarMacro);

    assertThat(fooBarMacroPiece.getTargets()).hasSize(1);
    assertThat(fooBarMacroPiece.getTargets(Rule.class)).hasSize(1);
    Target fooBarTest = fooBarMacroPiece.getTarget("foo_bar_test");
    assertThat(fooBarTest).isNotNull();
    assertThat(fooBarTest.getLabel()).isEqualTo(Label.parseCanonical("//test_pkg:foo_bar_test"));
    assertThat(fooBarTest.getRuleClass()).isEqualTo(FAUX_TEST_CLASS.getName());
    assertThat(fooBarTest.getPackageoid()).isSameInstanceAs(fooBarMacroPiece);
    assertThat(fooBarTest.getDeclaringMacro()).isSameInstanceAs(fooBarMacro);
    assertThat(fooBarTest.getDeclaringPackage()).isEqualTo(FAKE_BZL_LABEL.getPackageIdentifier());
  }

  private PackagePiece.ForBuildFile.Builder minimalBuildFilePieceBuilder(String name) {
    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(name);
    return PackagePiece.ForBuildFile.newBuilder(
            PackageSettings.DEFAULTS,
            new PackagePieceIdentifier.ForBuildFile(pkgId),
            /* filename= */ RootedPath.toRootedPath(
                Root.fromPath(fileSystem.getPath("/irrelevantRoot")),
                PathFragment.create(name + "/BUILD")),
            "workspace",
            /* associatedModuleName= */ Optional.empty(),
            /* associatedModuleVersion= */ Optional.empty(),
            /* noImplicitFileExport= */ true,
            /* simplifyUnconditionalSelectsInRuleAttrs= */ StarlarkSemantics.DEFAULT.getBool(
                BuildLanguageOptions.INCOMPATIBLE_SIMPLIFY_UNCONDITIONAL_SELECTS_IN_RULE_ATTRS),
            /* repositoryMapping= */ RepositoryMapping.EMPTY,
            /* mainRepositoryMapping= */ null,
            /* cpuBoundSemaphore= */ null,
            PackageOverheadEstimator.NOOP_ESTIMATOR,
            /* generatorMap= */ null,
            /* configSettingVisibilityPolicy= */ null,
            /* globber= */ null,
            /* enableNameConflictChecking= */ true,
            /* trackFullMacroInformation= */ false,
            Package.Builder.PackageLimits.DEFAULTS)
        .setLoads(ImmutableList.of());
  }

  private PackagePiece.ForMacro.Builder minimalMacroPieceBuilder(
      MacroInstance macro,
      PackagePieceIdentifier parentIdentifier,
      PackagePiece.ForBuildFile pieceForBuildFile) {
    return PackagePiece.ForMacro.newBuilder(
        pieceForBuildFile.getMetadata(),
        pieceForBuildFile.getDeclarations(),
        macro,
        parentIdentifier,
        /* simplifyUnconditionalSelectsInRuleAttrs= */ StarlarkSemantics.DEFAULT.getBool(
            BuildLanguageOptions.INCOMPATIBLE_SIMPLIFY_UNCONDITIONAL_SELECTS_IN_RULE_ATTRS),
        /* mainRepositoryMapping= */ null,
        /* cpuBoundSemaphore= */ null,
        PackageOverheadEstimator.NOOP_ESTIMATOR,
        /* enableNameConflictChecking= */ true,
        /* trackFullMacroInformation= */ false,
        Package.Builder.PackageLimits.DEFAULTS);
  }

  @CanIgnoreReturnValue
  private static Rule addRule(
      TargetDefinitionContext targetDefinitionContext, Label label, RuleClass ruleClass)
      throws Exception {
    Rule rule =
        targetDefinitionContext.createRule(
            label, ruleClass, /* threadCallStack= */ ImmutableList.of());
    rule.populateOutputFiles(
        new StoredEventHandler(), targetDefinitionContext.getPackageIdentifier());
    targetDefinitionContext.addRule(rule);
    return rule;
  }

  private MacroClass noopMacroClass(String name) {
    return new MacroClass.Builder(noopMacroImplementation)
        .setName(name)
        .setDefiningBzlLabel(FAKE_BZL_LABEL)
        .build();
  }

  private MacroClass failMacroClass(String name) {
    return new MacroClass.Builder(failMacroImplementation)
        .setName(name)
        .setDefiningBzlLabel(FAKE_BZL_LABEL)
        .build();
  }

  @CanIgnoreReturnValue
  private MacroInstance addMacro(
      TargetDefinitionContext targetDefinitionContext,
      MacroClass macroClass,
      String macroInstanceName)
      throws Exception {
    MacroInstance macro =
        targetDefinitionContext.createMacro(
            macroClass, macroInstanceName, /* sameNameDepth= */ 1, ImmutableList.of());
    macroClass
        .getAttributeProvider()
        .populateRuleAttributeValues(
            macro,
            targetDefinitionContext,
            new RuleFactory.BuildLangTypedAttributeValuesMap(
                ImmutableMap.of("name", macroInstanceName, "visibility", Starlark.NONE)),
            /* failOnUnknownAttributes= */ true,
            /* isStarlark= */ true);
    targetDefinitionContext.addMacro(macro);
    return macro;
  }
}
