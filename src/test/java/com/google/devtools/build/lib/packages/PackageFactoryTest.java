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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.PackageValidator.InvalidPackageException;
import com.google.devtools.build.lib.packages.util.PackageFactoryTestBase;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@code PackageFactory}.
 */
@RunWith(JUnit4.class)
public class PackageFactoryTest extends PackageFactoryTestBase {

  @Test
  public void testCreatePackage() throws Exception {
    Path buildFile = scratch.file("/pkgname/BUILD", "# empty build file ");
    Package pkg = packages.createPackage("pkgname", RootedPath.toRootedPath(root, buildFile));
    assertThat(pkg.getName()).isEqualTo("pkgname");
    assertThat(Sets.newHashSet(pkg.getTargets(Rule.class))).isEmpty();
  }

  @Test
  public void testBadRuleName() throws Exception {
    events.setFailFast(false);

    Path buildFile = scratch.file("/badrulename/BUILD", "cc_library(name = 3)");
    Package pkg = packages.createPackage("badrulename", RootedPath.toRootedPath(root, buildFile));

    events.assertContainsError("cc_library 'name' attribute must be a string");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testNoRuleName() throws Exception {
    events.setFailFast(false);

    Path buildFile = scratch.file("/badrulename/BUILD", "cc_library()");
    Package pkg = packages.createPackage("badrulename", RootedPath.toRootedPath(root, buildFile));

    events.assertContainsError("cc_library rule has no 'name' attribute");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testBadPackageName() throws Exception {
    NoSuchPackageException e =
        assertThrows(
            NoSuchPackageException.class,
            () ->
                packages.createPackage(
                    "not even a legal/.../label",
                    RootedPath.toRootedPath(root, emptyBuildFile("not even a legal/.../label"))));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "no such package 'not even a legal/.../label': "
                + "illegal package name: 'not even a legal/.../label' ");
  }

  @Test
  public void testColonInExportsFilesTargetName() throws Exception {
    events.setFailFast(false);
    Path path =
        scratch.file(
            "/googledata/cafe/BUILD",
            "exports_files(['houseads/house_ads:ca-aol_parenting_html'])");
    Package pkg = packages.createPackage("googledata/cafe", RootedPath.toRootedPath(root, path));
    events.assertContainsError("target names may not contain ':'");
    assertThat(pkg.getTargets(FileTarget.class).toString())
        .doesNotContain("houseads/house_ads:ca-aol_parenting_html");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testExportsFilesVisibilityMustBeSequence() throws Exception {
    expectEvalError(
        "in call to exports_files(), parameter 'visibility' got value of type 'depset', want"
            + " 'sequence or NoneType'",
        "exports_files(srcs=[], visibility=depset(['notice']))");
  }

  @Test
  public void testExportsFilesLicensesMustBeSequence() throws Exception {
    expectEvalError(
        "in call to exports_files(), parameter 'licenses' got value of type 'depset', want"
            + " 'sequence of strings or NoneType'",
        "exports_files(srcs=[], licenses=depset(['notice']))");
  }

  @Test
  public void testPackageNameWithPROTECTEDIsOk() throws Exception {
    events.setFailFast(false);
    // One "PROTECTED":
    assertThat(isValidPackageName("foo/PROTECTED/bar")).isTrue();
    // Multiple "PROTECTED"s:
    assertThat(isValidPackageName("foo/PROTECTED/bar/PROTECTED/wiz")).isTrue();
  }

  @Test
  public void testDuplicatedDependencies() throws Exception {
    events.setFailFast(false);
    Path buildFile =
        scratch.file(
            "/has_dupe/BUILD",
            "cc_library(name='dep')",
            "cc_library(name='has_dupe', deps=[':dep', ':dep'])");

    Package pkg = packages.createPackage("has_dupe", RootedPath.toRootedPath(root, buildFile));
    events.assertContainsError(
        "Label '//has_dupe:dep' is duplicated in the 'deps' " + "attribute of rule 'has_dupe'");
    assertThat(pkg.containsErrors()).isTrue();
    assertThat(pkg.getRule("has_dupe")).isNotNull();
    assertThat(pkg.getRule("dep")).isNotNull();
    assertThat(pkg.getRule("has_dupe").containsErrors()).isTrue();
    assertThat(pkg.getRule("dep").containsErrors()).isTrue(); // because all rules in an
    // errant package are
    // themselves errant.
  }

  @Test
  public void testPrefixWithinSameRule1() throws Exception {
    events.setFailFast(false);
    Path buildFile =
        scratch.file(
            "/fruit/orange/BUILD",
            "genrule(name='orange', srcs=[], outs=['a', 'a/b'], cmd='')");

    packages.createPackage("fruit/orange", RootedPath.toRootedPath(root, buildFile));
    events.assertContainsError("rule 'orange' has conflicting output files 'a/b' and 'a");
  }

  @Test
  public void testPrefixWithinSameRule2() throws Exception {
    events.setFailFast(false);
    Path buildFile =
        scratch.file(
            "/fruit/orange/BUILD",
            "genrule(name='orange', srcs=[], outs=['a/b', 'a'], cmd='')");

    packages.createPackage("fruit/orange", RootedPath.toRootedPath(root, buildFile));
    events.assertContainsError("rule 'orange' has conflicting output files 'a' and 'a/b");
  }

  @Test
  public void testPrefixBetweenRules1() throws Exception {
    events.setFailFast(false);
    Path buildFile =
        scratch.file(
            "/fruit/kiwi/BUILD",
            "genrule(name='kiwi1', srcs=[], outs=['a'], cmd='')",
            "genrule(name='kiwi2', srcs=[], outs=['a/b'], cmd='')");
    packages.createPackage("fruit/kiwi", RootedPath.toRootedPath(root, buildFile));
    events.assertContainsError(
        "output file 'a/b' of rule 'kiwi2' conflicts " + "with output file 'a' of rule 'kiwi1'");
  }

  @Test
  public void testPrefixBetweenRules2() throws Exception {
    events.setFailFast(false);
    Path buildFile =
        scratch.file(
            "/fruit/kiwi/BUILD",
            "genrule(name='kiwi1', srcs=[], outs=['a/b'], cmd='')",
            "genrule(name='kiwi2', srcs=[], outs=['a'], cmd='')");
    packages.createPackage("fruit/kiwi", RootedPath.toRootedPath(root, buildFile));
    events.assertContainsError(
        "output file 'a' of rule 'kiwi2' conflicts " + "with output file 'a/b' of rule 'kiwi1'");
  }

  @Test
  public void testPackageConstantIsForbidden() throws Exception {
    events.setFailFast(false);
    Path buildFile = scratch.file("/pina/BUILD", "cc_library(name=PACKAGE_NAME + '-colada')");
    packages.createPackage("pina", RootedPath.toRootedPath(root, buildFile));
    events.assertContainsError("The value 'PACKAGE_NAME' has been removed");
  }

  @Test
  public void testPackageNameFunction() throws Exception {
    Path buildFile = scratch.file("/pina/BUILD", "cc_library(name=package_name() + '-colada')");

    Package pkg = packages.createPackage("pina", RootedPath.toRootedPath(root, buildFile));
    events.assertNoWarningsOrErrors();
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getRule("pina-colada")).isNotNull();
    assertThat(pkg.getRule("pina-colada").containsErrors()).isFalse();
    assertThat(Sets.newHashSet(pkg.getTargets(Rule.class)).size()).isSameInstanceAs(1);
  }

  @Test
  public void testPackageConstantInExternalRepositoryIsForbidden() throws Exception {
    events.setFailFast(false);
    Path buildFile =
        scratch.file(
            "/external/a/b/BUILD", "genrule(name='c', srcs=[], outs=['ao'], cmd=REPOSITORY_NAME)");
    packages.createPackage(
        PackageIdentifier.create("@a", PathFragment.create("b")),
        RootedPath.toRootedPath(root, buildFile),
        events.reporter());
    events.assertContainsError("The value 'REPOSITORY_NAME' has been removed");
  }

  @Test
  public void testPackageFunctionInExternalRepository() throws Exception {
    Path buildFile =
        scratch.file(
            "/external/a/b/BUILD",
            "genrule(name='c', srcs=[], outs=['o'], cmd=repository_name() + ' ' + package_name())");
    Package pkg =
        packages.createPackage(
            PackageIdentifier.create("@a", PathFragment.create("b")),
            RootedPath.toRootedPath(root, buildFile),
            events.reporter());
    Rule c = pkg.getRule("c");
    assertThat(AggregatingAttributeMapper.of(c).get("cmd", Type.STRING)).isEqualTo("@a b");
  }

  @Test
  public void testDuplicateRuleName() throws Exception {
    events.setFailFast(false);

    Path buildFile =
        scratch.file(
            "/duplicaterulename/BUILD",
            "proto_library(name = 'spellcheck_proto',",
            "         srcs = ['spellcheck.proto'],",
            "         cc_api_version = 2)",
            "cc_library(name = 'spellcheck_proto')", // conflict error stops execution
            "x = 1//0"); // not reached
    Package pkg =
        packages.createPackage("duplicaterulename", RootedPath.toRootedPath(root, buildFile));
    events.assertContainsError(
        "cc_library rule 'spellcheck_proto' in package 'duplicaterulename' conflicts with existing"
            + " proto_library rule");
    events.assertDoesNotContainEvent("division by zero");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testBuildFileTargetExists() throws Exception {
    Path buildFile = scratch.file("/foo/BUILD", "");
    Package pkg = packages.createPackage("foo", RootedPath.toRootedPath(root, buildFile));

    Target target = pkg.getTarget("BUILD");
    assertThat(target.getName()).isEqualTo("BUILD");

    // Test that it's memoized:
    assertThat(pkg.getTarget("BUILD")).isSameInstanceAs(target);
  }

  @Test
  public void testCreationOfInputFiles() throws Exception {
    Path buildFile =
        scratch.file(
            "/foo/BUILD",
            "exports_files(['Z'])",
            "cc_library(name='W', deps=['X', 'Y'])",
            "cc_library(name='X', srcs=['X'])",
            "cc_library(name='Y')");
    Package pkg = packages.createPackage("foo", RootedPath.toRootedPath(root, buildFile));
    assertThat(pkg.containsErrors()).isFalse();

    // X is a rule with a circular self-dependency.
    assertThat(pkg.getTarget("X").getClass()).isSameInstanceAs(Rule.class);

    // Y is a rule
    assertThat(pkg.getTarget("Y").getClass()).isSameInstanceAs(Rule.class);

    // Z is a file
    assertThat(pkg.getTarget("Z").getClass()).isSameInstanceAs(InputFile.class);

    // A is nothing
    NoSuchTargetException e = assertThrows(NoSuchTargetException.class, () -> pkg.getTarget("A"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "no such target '//foo:A': "
                + "target 'A' not declared in package 'foo' defined by /foo/BUILD");

    // These are the only input files: BUILD, Z
    Set<String> inputFiles = Sets.newTreeSet();
    for (InputFile inputFile : pkg.getTargets(InputFile.class)) {
      inputFiles.add(inputFile.getName());
    }
    assertThat(Lists.newArrayList(inputFiles)).containsExactly("BUILD", "Z").inOrder();
  }

  @Test
  public void testDuplicateRuleIsNotAddedToPackage() throws Exception {
    events.setFailFast(false);
    Path path =
        scratch.file(
            "/dup/BUILD",
            "proto_library(name = 'dup_proto',",
            "              srcs  = ['dup.proto'],",
            "              cc_api_version = 2)",
            "",
            "cc_library(name = 'dup_proto',",
            "           srcs = ['dup.pb.cc', 'dup.pb.h'])");
    Package pkg = packages.createPackage("dup", RootedPath.toRootedPath(root, path));
    events.assertContainsError(
        "cc_library rule 'dup_proto' in package 'dup' "
            + "conflicts with existing proto_library rule");
    assertThat(pkg.containsErrors()).isTrue();

    Rule dupProto = pkg.getRule("dup_proto");
    // Check that the first rule of the given name "wins", and that each of the
    // "winning" rule's outputs is a member of the package.
    assertThat(dupProto.getRuleClass()).isEqualTo("proto_library");
    for (OutputFile out : dupProto.getOutputFiles()) {
      assertThat(pkg.getTargets(FileTarget.class)).contains(out);
    }
  }

  @Test
  public void testConflictingRuleDoesNotUpdatePackage() throws Exception {
    events.setFailFast(false);

    // In this test, rule2's outputs conflict with rule1, so rule2 is rejected.
    // However, we must check that neither rule2, nor any of its inputs or
    // outputs is a member of the package, and that the conflicting output file
    // "out2" still has rule1 as its getGeneratingRule().
    Path path =
        scratch.file(
            "/conflict/BUILD",
            "genrule(name = 'rule1',",
            "        cmd = '',",
            "        srcs = ['in1', 'in2'],",
            "        outs = ['out1', 'out2'])",
            "genrule(name = 'rule2',",
            "        cmd = '',",
            "        srcs = ['in3', 'in4'],",
            "        outs = ['out3', 'out2'])");
    Package pkg = packages.createPackage("conflict", RootedPath.toRootedPath(root, path));
    events.assertContainsError(
        "generated file 'out2' in rule 'rule2' "
            + "conflicts with existing generated file from rule 'rule1'");
    assertThat(pkg.containsErrors()).isTrue();

    assertThat(pkg.getRule("rule2")).isNull();

    // Ensure that rule2's "out2" didn't overwrite rule1's:
    assertThat(((OutputFile) pkg.getTarget("out2")).getGeneratingRule())
        .isSameInstanceAs(pkg.getRule("rule1"));

    // None of rule2, its inputs, or its outputs should belong to pkg:
    List<Target> found = new ArrayList<>();
    for (String targetName : ImmutableList.of("rule2", "in3", "in4", "out3")) {
      try {
        found.add(pkg.getTarget(targetName));
        // No fail() here: if there's no exception, we add the name to a list
        // and we check below that it's empty.
      } catch (NoSuchTargetException e) {
        /* good! */
      }
    }
    assertThat(found).isEmpty();
  }

  // Was: Regression test for bug "Rules declared after an error in
  // a package should be considered 'in error'".
  // Then: Regression test for bug "Why aren't ERRORS considered
  // fatal?*"
  // Now: Regression test for: execution should stop at the first EvalException;
  // all rules created prior to the exception error are marked in error.
  @Test
  public void testAllRulesInErrantPackageAreInError() throws Exception {
    events.setFailFast(false);
    Path path =
        scratch.file(
            "/error/BUILD",
            "genrule(name = 'rule1',",
            "        cmd = ':',",
            "        outs = ['out.1'])",
            "list = ['bad']",
            "x = 1//0", // dynamic error
            "genrule(name = 'rule2',",
            "        cmd = ':',",
            "        outs = list)");
    Package pkg = packages.createPackage("error", RootedPath.toRootedPath(root, path));
    events.assertContainsError("division by zero");

    assertThat(pkg.containsErrors()).isTrue();

    // rule1 would be fine but is still marked as in error:
    assertThat(pkg.getRule("rule1").containsErrors()).isTrue();

    // rule2's genrule is never executed.
    Rule rule2 = pkg.getRule("rule2");
    assertThat(rule2).isNull();
  }

  @Test
  public void testHelpfulErrorForMissingExportsFiles() throws Exception {
    Path path = scratch.file("/x/BUILD", "cc_library(name='x', srcs=['x.cc'])");
    scratch.file("/x/x.cc");
    scratch.file("/x/y.cc");
    scratch.file("/x/dir/dummy");

    Package pkg = packages.createPackage("x", RootedPath.toRootedPath(root, path));

    assertThat(pkg.getTarget("x.cc")).isNotNull(); // existing and mentioned.

    NoSuchTargetException e =
        assertThrows(NoSuchTargetException.class, () -> pkg.getTarget("y.cc"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "no such target '//x:y.cc': "
                + "target 'y.cc' not declared in package 'x'; "
                + "however, a source file of this name exists.  "
                + "(Perhaps add 'exports_files([\"y.cc\"])' to x/BUILD?) "
                + "defined by /x/BUILD");

    e = assertThrows(NoSuchTargetException.class, () -> pkg.getTarget("z.cc"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "no such target '//x:z.cc': "
                + "target 'z.cc' not declared in package 'x' (did you mean 'x.cc'?) "
                + "defined by /x/BUILD");

    e = assertThrows(NoSuchTargetException.class, () -> pkg.getTarget("dir"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "no such target '//x:dir': target 'dir' not declared in package 'x'; "
                + "however, a source directory of this name exists.  "
                + "(Perhaps add 'exports_files([\"dir\"])' to x/BUILD, "
                + "or define a filegroup?) defined by /x/BUILD");
  }

  @Test
  public void testTestSuitesImplicitlyDependOnAllRulesInPackage() throws Exception {
    Path path =
        scratch.file(
            "/x/BUILD",
            "java_test(name='j')",
            "test_suite(name='t1')",
            "test_suite(name='t2', tests=['//foo'])",
            "test_suite(name='t3', tests=['//foo'])",
            "cc_test(name='c')");
    Package pkg = packages.createPackage("x", RootedPath.toRootedPath(root, path));

    // Things to note:
    // - the t1 refers to both :j and :c, even though :c is a forward reference.
    // - $implicit_tests is empty unless tests=[]

    assertThat(attributes(pkg.getRule("t1")).get("$implicit_tests", BuildType.LABEL_LIST))
        .containsExactlyElementsIn(
            Sets.newHashSet(
                Label.parseAbsolute("//x:c", ImmutableMap.of()),
                Label.parseAbsolute("//x:j", ImmutableMap.of())));
    assertThat(attributes(pkg.getRule("t2")).get("$implicit_tests", BuildType.LABEL_LIST))
        .isEmpty();
    assertThat(attributes(pkg.getRule("t3")).get("$implicit_tests", BuildType.LABEL_LIST))
        .isEmpty();
  }

  @Test
  public void testPackageValidationFailureRegisteredAfterLoading() throws Exception {
    Path path = scratch.file("/x/BUILD", "sh_library(name='y')");

    dummyPackageValidator.setImpl(
        (pkg, eventHandler) -> {
          if (pkg.getName().equals("x")) {
            eventHandler.handle(Event.warn("warning event"));
            throw new InvalidPackageException(pkg.getPackageIdentifier(), "nope");
          }
        });

    Package pkg = packages.createPackage("x", RootedPath.toRootedPath(root, path));
    assertThat(pkg.containsErrors()).isFalse();

    StoredEventHandler eventHandler = new StoredEventHandler();
    InvalidPackageException expected =
        assertThrows(
            InvalidPackageException.class,
            () ->
                packages
                    .factory()
                    .afterDoneLoadingPackage(
                        pkg,
                        StarlarkSemantics.DEFAULT_SEMANTICS,
                        /*loadTimeNanos=*/ 0,
                        eventHandler));
    assertThat(expected).hasMessageThat().contains("no such package 'x': nope");
    assertThat(eventHandler.getEvents()).containsExactly(Event.warn("warning event"));
  }

  @Test
  public void testGlobDirectoryExclusion() throws Exception {
    emptyFile("/fruit/data/apple");
    emptyFile("/fruit/data/pear");
    emptyFile("/fruit/data/berry/black");
    emptyFile("/fruit/data/berry/blue");
    Path file =
        scratch.file(
            "/fruit/BUILD",
            "cc_library(name = 'yes', srcs = glob(['data/*']))",
            "cc_library(name = 'no',  srcs = glob(['data/*'], exclude_directories=0))");
    Package pkg = packages.eval("fruit", RootedPath.toRootedPath(root, file));
    events.assertNoWarningsOrErrors();
    List<Label> yesFiles = attributes(pkg.getRule("yes")).get("srcs", BuildType.LABEL_LIST);
    List<Label> noFiles = attributes(pkg.getRule("no")).get("srcs", BuildType.LABEL_LIST);

    assertThat(yesFiles)
        .containsExactly(
            Label.parseAbsolute("@//fruit:data/apple", ImmutableMap.of()),
            Label.parseAbsolute("@//fruit:data/pear", ImmutableMap.of()));

    assertThat(noFiles)
        .containsExactly(
            Label.parseAbsolute("@//fruit:data/apple", ImmutableMap.of()),
            Label.parseAbsolute("@//fruit:data/pear", ImmutableMap.of()),
            Label.parseAbsolute("@//fruit:data/berry", ImmutableMap.of()));
  }

  // TODO(bazel-team): This is really a test for GlobCache.
  @Test
  public void testRecursiveGlob() throws Exception {
    emptyFile("/rg/a.cc");
    emptyFile("/rg/foo/bar.cc");
    emptyFile("/rg/foo/foo.cc");
    emptyFile("/rg/foo/wiz/bam.cc");
    emptyFile("/rg/foo/wiz/bum.cc");
    emptyFile("/rg/foo/wiz/quid/gav.cc");
    Path file =
        scratch.file(
            "/rg/BUILD",
            "cc_library(name = 'ri', srcs = glob(['**/*.cc']))",
            "cc_library(name = 're', srcs = glob(['*.cc'], exclude=['**/*.c']))");
    Package pkg = packages.eval("rg", RootedPath.toRootedPath(root, file));
    events.assertNoWarningsOrErrors();

    assertEvaluates(
        pkg,
        ImmutableList.of(
            "BUILD",
            "a.cc",
            "foo",
            "foo/bar.cc",
            "foo/foo.cc",
            "foo/wiz",
            "foo/wiz/bam.cc",
            "foo/wiz/bum.cc",
            "foo/wiz/quid",
            "foo/wiz/quid/gav.cc"),
        "**");

    assertEvaluates(
        pkg,
        ImmutableList.of(
            "a.cc",
            "foo/bar.cc",
            "foo/foo.cc",
            "foo/wiz/bam.cc",
            "foo/wiz/bum.cc",
            "foo/wiz/quid/gav.cc"),
        "**/*.cc");
    assertEvaluates(
        pkg, ImmutableList.of("foo/bar.cc", "foo/wiz/bam.cc", "foo/wiz/bum.cc"), "**/b*.cc");
    assertEvaluates(
        pkg,
        ImmutableList.of(
            "foo/bar.cc", "foo/foo.cc", "foo/wiz/bam.cc", "foo/wiz/bum.cc", "foo/wiz/quid/gav.cc"),
        "**/*/*.cc");
    assertEvaluates(pkg, ImmutableList.of("foo/wiz/quid/gav.cc"), "foo/**/quid/*.cc");

    assertEvaluates(
        pkg,
        Collections.<String>emptyList(),
        ImmutableList.of("*.cc", "*/*.cc", "*/*/*.cc"),
        ImmutableList.of("**/*.cc"));
    assertEvaluates(
        pkg,
        Collections.<String>emptyList(),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("**/*.cc"));
    assertEvaluates(
        pkg,
        Collections.<String>emptyList(),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("*.cc", "*/*.cc", "*/*/*.cc", "*/*/*/*.cc"));
    assertEvaluates(
        pkg,
        Collections.<String>emptyList(),
        ImmutableList.of("**"),
        ImmutableList.of("*", "*/*", "*/*/*", "*/*/*/*"));
    assertEvaluates(
        pkg,
        ImmutableList.of(
            "foo/bar.cc", "foo/foo.cc", "foo/wiz/bam.cc", "foo/wiz/bum.cc", "foo/wiz/quid/gav.cc"),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("*.cc"));
    assertEvaluates(
        pkg,
        ImmutableList.of("a.cc", "foo/wiz/bam.cc", "foo/wiz/bum.cc", "foo/wiz/quid/gav.cc"),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("*/*.cc"));
    assertEvaluates(
        pkg,
        ImmutableList.of("a.cc", "foo/bar.cc", "foo/foo.cc", "foo/wiz/quid/gav.cc"),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("**/wiz/*.cc"));
  }

  @Test
  public void testTooManyArgumentsGlobErrors() throws Exception {
    events.setFailFast(false);
    assertGlobFails(
        "glob(['incl'],['excl'],3,True,'extraarg')",
        "glob() accepts no more than 4 positional arguments but got 5");
  }

  @Test
  public void testGlobEnforcesListArgument() throws Exception {
    events.setFailFast(false);
    assertGlobFails(
        "glob(1, exclude=2)",
        "in call to glob(), parameter 'include' got value of type 'int', want 'sequence of"
            + " strings'");
  }

  @Test
  public void testGlobEnforcesListOfStringsArguments() throws Exception {
    events.setFailFast(false);
    assertGlobFails(
        "glob(['a', 'b'], exclude=['c', 42])",
        "expected value of type 'string' for element 1 of 'glob' argument, but got 42 (int)");
  }

  @Test
  public void testGlobNegativeTest() throws Exception {
    // Negative test that assertGlob does throw an error when asserting against the wrong values.
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                assertGlobMatches(
                    /*result=*/ ImmutableList.of("Wombat1.java", "This_file_doesn_t_exist.java"),
                    /*includes=*/ ImmutableList.of("W*", "subdir"),
                    /*excludes=*/ ImmutableList.<String>of(),
                    /* excludeDirs= */ true));
    assertThat(e).hasMessageThat().isEqualTo("ERROR /globs/BUILD:2:73: incorrect glob result");
  }

  @Test
  public void testGlobExcludeDirectories() throws Exception {
    assertGlobMatches(
        /*result=*/ ImmutableList.of("Wombat1.java", "Wombat2.java"),
        /*includes=*/ ImmutableList.of("W*", "subdir"),
        /*excludes=*/ ImmutableList.<String>of(),
        /* excludeDirs= */ true);
  }

  @Test
  public void testGlobDoesNotExcludeDirectories() throws Exception {
    assertGlobMatches(
        /*result=*/ ImmutableList.of("Wombat1.java", "Wombat2.java", "subdir"),
        /*includes=*/ ImmutableList.of("W*", "subdir"),
        /*excludes=*/ ImmutableList.<String>of(),
        /* excludeDirs= */ false);
  }

  @Test
  public void testGlobWithEmptyExcludedList() throws Exception {
    assertGlobMatches(
        /*result=*/ ImmutableList.of("Wombat1.java", "Wombat2.java"),
        /*includes=*/ ImmutableList.of("W*"),
        /*excludes=*/ Collections.<String>emptyList(),
        /* excludeDirs= */ false);
  }

  @Test
  public void testGlobWithQuestionMarkProducesError() throws Exception {
    assertGlobProducesError("Wombat?.java", true);
  }

  @Test
  public void testGlobWithoutQuestionMarkDoesntProduceError() throws Exception {
    assertGlobProducesError("Wombat*.java", false);
  }

  @Test
  public void testGlobWithNonMatchingExcludedList() throws Exception {
    assertGlobMatches(
        /*result=*/ ImmutableList.of("Wombat1.java"),
        /*includes=*/ ImmutableList.of("W*"),
        /*excludes=*/ ImmutableList.of("*2*"),
        /* excludeDirs= */ false);
  }

  @Test
  public void testGlobWithTwoMatchingGlobExpressionsAndNonmatchingExclusion() throws Exception {
    assertGlobMatches(
        /*result=*/ ImmutableList.of("Wombat1.java", "subdir/Wombat3.java"),
        /*includes=*/ ImmutableList.of("W*", "subdir/W*"),
        /*excludes=*/ ImmutableList.of("*2*"),
        /* excludeDirs= */ false);
  }

  @Test
  public void testGlobWithSubdirMatchAndExclusion() throws Exception {
    assertGlobMatches(
        /*result=*/ ImmutableList.of("subdir/Wombat3.java"),
        /*includes=*/ ImmutableList.of("W*", "subdir/W*"),
        /*excludes=*/ ImmutableList.of("Wombat*.java"),
        /* excludeDirs= */ false);
  }

  @Test
  public void testBadCharacterInGlob() throws Exception {
   events.setFailFast(false);
   assertGlobFails("glob(['?'])", "glob pattern '?' contains forbidden '?' wildcard");
  }

  @Test
  public void testBadExcludePattern() throws Exception {
    events.setFailFast(false);
    // The 'exclude' check is currently only reached if the pattern is "complex".
    // This seems like a bug:
    //   assertGlobFails("glob(['BUILD'], ['/'])", "pattern cannot be absolute");
    assertGlobFails("glob(['BUILD'], ['/*/*'])", "pattern cannot be absolute");
  }

  @Test
  public void testGlobEscapesAt() throws Exception {
    // See lib.skyframe.PackageFunctionTest.globEscapesAt and
    // https://github.com/bazelbuild/bazel/issues/10606.
    scratch.file("/p/@f.txt");
    Path file = scratch.file("/p/BUILD", "print(glob(['*.txt'])[0])");
    events.setFailFast(false); // we need this to use print (!)
    packages.eval("p", RootedPath.toRootedPath(root, file));
    events.assertNoWarningsOrErrors();
    events.assertContainsDebug(":@f.txt"); // observe prepended colon
  }

  /**
   * Tests that a glob evaluation that encounters an I/O error throws instead of constructing a
   * package.
   */
  @Test
  public void testGlobWithIOErrors() throws Exception {
    events.setFailFast(false);

    scratch.dir("/pkg");
    scratch.dir("/pkg/globs");
    Path unreadableSubdir = scratch.resolve("/pkg/globs/unreadable_subdir");
    unreadableSubdir.createDirectory();
    unreadableSubdir.setReadable(false);

    Path file = scratch.file("/pkg/BUILD", "cc_library(name = 'c', srcs = glob(['globs/**']))");
    assertThrows(
        NoSuchPackageException.class,
        () -> packages.eval("pkg", RootedPath.toRootedPath(root, file)));
    events.assertContainsError("Directory is not readable");
  }

  @Test
  public void testNativeModuleIsDisabled() throws Exception {
    events.setFailFast(false);
    Path buildFile = scratch.file("/pkg/BUILD", "native.cc_library(name='bar')");
    Package pkg = packages.createPackage("pkg", RootedPath.toRootedPath(root, buildFile));
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testPackageGroupSpecMinimal() throws Exception {
    expectEvalSuccess("package_group(name='skin', packages=[])");
  }

  @Test
  public void testPackageGroupSpecSimple() throws Exception {
    expectEvalSuccess("package_group(name='skin', packages=['//group/abelian'])");
  }

  @Test
  public void testPackageGroupSpecEmpty() throws Exception {
    expectEvalSuccess("package_group(name='seed')");
  }

  @Test
  public void testPackageGroupSpecIncludes() throws Exception {
    expectEvalSuccess(
        "package_group(name='wine',",
        "              includes=['//wine:cabernet_sauvignon',",
        "                        '//wine:pinot_noir'])");
  }

  @Test
  public void testPackageGroupSpecBad() throws Exception {
    expectEvalError("invalid package name", "package_group(name='skin', packages=['--25:17--'])");
  }

  @Test
  public void testPackageGroupsWithSameName() throws Exception {
    expectEvalError(
        "conflicts with existing package group",
        "package_group(name='skin', packages=[])",
        "package_group(name='skin', packages=[])");
  }

  @Test
  public void testPackageGroupNamedArguments() throws Exception {
    expectEvalError(
        "package_group() got unexpected positional argument", "package_group('skin', name = 'x')");
  }

  @Test
  public void testPackageSpecMinimal() throws Exception {
    Package pkg = expectEvalSuccess("package(default_visibility=[])");
    assertThat(pkg.getDefaultVisibility()).isNotNull();
  }

  @Test
  public void testPackageSpecSimple() throws Exception {
    expectEvalSuccess("package(default_visibility=['//group:lie'])");
  }

  @Test
  public void testPackageSpecBad() throws Exception {
    expectEvalError("invalid target name", "package(default_visibility=[':::'])");
  }

  @Test
  public void testDoublePackageSpecification() throws Exception {
    expectEvalError(
        "can only be used once",
        "package(default_visibility=[])",
        "package(default_visibility=[])");
  }

  @Test
  public void testEmptyPackageSpecification() throws Exception {
    expectEvalError("at least one argument must be given to the 'package' function", "package()");
  }

  @Test
  public void testDefaultTestonly() throws Exception {
    Package pkg = expectEvalSuccess("package(default_testonly = 1)");
    assertThat(pkg.getDefaultTestOnly()).isTrue();
  }

  @Test
  public void testDefaultDeprecation() throws Exception {
    String testMessage = "OMG PONIES!";
    Package pkg = expectEvalSuccess("package(default_deprecation = \"" + testMessage + "\")");
    assertThat(pkg.getDefaultDeprecation()).isEqualTo(testMessage);
  }

  @Test
  public void testExportsBuildFile() throws Exception {
    Package pkg =
        expectEvalSuccess("exports_files(['BUILD'], visibility=['//visibility:private'])");
    assertThat(pkg.getTarget("BUILD")).isEqualTo(pkg.getBuildFile());
  }

  @Test
  public void testDefaultDeprecationPropagation() throws Exception {
    String msg = "I am completely operational, and all my circuits are functioning perfectly.";
    Path file =
        scratch.file(
            "/foo/BUILD",
            "package(default_deprecation = \"" + msg + "\")",
            "sh_library(name = 'bar', srcs=['b'])");
    Package pkg = packages.eval("foo", RootedPath.toRootedPath(root, file));

    Rule fooRule = (Rule) pkg.getTarget("bar");
    String deprAttr =
        attributes(fooRule).get("deprecation", com.google.devtools.build.lib.packages.Type.STRING);
    assertThat(deprAttr).isEqualTo(msg);
  }

  @Test
  public void testDefaultTestonlyPropagation() throws Exception {
    Path file =
        scratch.file(
            "/foo/BUILD",
            "package(default_testonly = 1)",
            "sh_library(name = 'foo', srcs=['b'])",
            "sh_library(name = 'bar', srcs=['b'], testonly = 0)");
    Package pkg = packages.eval("foo", RootedPath.toRootedPath(root, file));

    Rule fooRule = (Rule) pkg.getTarget("foo");
    assertThat(
            attributes(fooRule)
                .get("testonly", com.google.devtools.build.lib.packages.Type.BOOLEAN))
        .isTrue();

    Rule barRule = (Rule) pkg.getTarget("bar");
    assertThat(
            attributes(barRule)
                .get("testonly", com.google.devtools.build.lib.packages.Type.BOOLEAN))
        .isFalse();
  }

  @Test
  public void testDefaultDeprecationOverriding() throws Exception {
    String msg = "I am completely operational, and all my circuits are functioning perfectly.";
    String deceive = "OMG PONIES!";
    Path file =
        scratch.file(
            "/foo/BUILD",
            "package(default_deprecation = \"" + deceive + "\")",
            "sh_library(name = 'bar', srcs=['b'], deprecation = \"" + msg + "\")");
    Package pkg = packages.eval("foo", RootedPath.toRootedPath(root, file));

    Rule fooRule = (Rule) pkg.getTarget("bar");
    String deprAttr =
        attributes(fooRule).get("deprecation", com.google.devtools.build.lib.packages.Type.STRING);
    assertThat(deprAttr).isEqualTo(msg);
  }

  @Test
  public void testPackageFeatures() throws Exception {
    Path file =
        scratch.file(
            "/a/BUILD",
            "sh_library(name='before')",
            "package(features=['b', 'c'])",
            "sh_library(name='after')");
    Package pkg = packages.eval("a", RootedPath.toRootedPath(root, file));

    assertThat(pkg.getFeatures()).containsExactly("b", "c");
  }

  @Test
  public void testTransientErrorsInGlobbing() throws Exception {
    events.setFailFast(false);
    Path buildFile =
        scratch.file("/e/BUILD", "sh_library(name = 'e', data = glob(['*.txt']))");
    Path parentDir = buildFile.getParentDirectory();
    scratch.file("/e/data.txt");
    throwOnReaddir = parentDir;
    assertThrows(
        NoSuchPackageException.class,
        () -> packages.createPackage("e", RootedPath.toRootedPath(root, buildFile)));
    events.setFailFast(true);
    throwOnReaddir = null;
    Package pkg = packages.createPackage("e", RootedPath.toRootedPath(root, buildFile));
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getRule("e")).isNotNull();
    List<?> globList = (List) pkg.getRule("e").getAttributeContainer().getAttr("data");
    assertThat(globList).containsExactly(Label.parseAbsolute("//e:data.txt", ImmutableMap.of()));
  }

  @Test
  public void testExportTwicePublicOK() throws Exception {
    // In theory, this could be an error, but too many existing files rely on it
    // and it is okay.
    expectEvalSuccess(
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:public\" ])",
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:public\" ])");
  }

  @Test
  public void testExportTwicePublicOK2() throws Exception {
    expectEvalSuccess(
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:private\" ])",
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:private\" ])");
  }

  @Test
  public void testExportTwiceFail() throws Exception {
    expectEvalError(
        "visibility for exported file 'a.cc' declared twice",
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:private\" ])",
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:public\" ])");
  }

  @Test
  public void testExportTwiceFail2() throws Exception {
    expectEvalError(
        "visibility for exported file 'a.cc' declared twice",
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:public\" ])",
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:private\" ])");
  }

  @Test
  public void testExportLicenseTwice() throws Exception {
    expectEvalError(
        "licenses for exported file 'a.cc' declared twice",
        "exports_files([\"a.cc\"], licenses = [\"notice\"])",
        "exports_files([\"a.cc\"], licenses = [\"notice\"])");
  }

  @Test
  public void testExportGenruleConflict() throws Exception {
    expectEvalError(
        "generated file 'a.cc' in rule 'foo' conflicts with existing source file",
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:public\" ])",
        "genrule(name = 'foo',",
        "    outs = ['a.cc'],",
        "    cmd = '')");
  }

  @Test
  public void testGenruleExportConflict() throws Exception {
    expectEvalError(
        "generated label '//pkg:a.cc' conflicts with existing generated file",
        "genrule(name = 'foo',",
        "    outs = ['a.cc'],",
        "    cmd = '')",
        "exports_files([\"a.cc\"],",
        "    visibility = [ \"//visibility:public\" ])");
  }

  @Test
  public void testValidEnvironmentGroup() throws Exception {
    expectEvalSuccess(
        "environment(name = 'foo')",
        "environment_group(name='group', environments = [':foo'], defaults = [':foo'])");
  }

  @Test
  public void testIncompleteEnvironmentGroup() throws Exception {
    expectEvalError(
        "environment_group() missing 1 required named argument: defaults",
        "environment(name = 'foo')",
        "environment_group(name='group', environments = [':foo'])");
  }

  @Test
  public void testEnvironmentGroupMissingTarget() throws Exception {
    expectEvalError(
        "environment //pkg:foo does not exist",
        "environment_group(name='group', environments = [':foo'], defaults = [':foo'])");
  }

  @Test
  public void testEnvironmentGroupWrongTargetType() throws Exception {
    expectEvalError(
        "//pkg:foo is not a valid environment",
        "cc_library(name = 'foo')",
        "environment_group(name='group', environments = [':foo'], defaults = [':foo'])");
  }

  @Test
  public void testEnvironmentGroupWrongPackage() throws Exception {
    expectEvalError(
        "//foo:foo is not in the same package as group //pkg:group",
        "environment_group(name='group', environments = ['//foo'], defaults = ['//foo'])");
  }

  @Test
  public void testEnvironmentGroupInvalidDefault() throws Exception {
    expectEvalError(
        "default //pkg:bar is not a declared environment for group //pkg:group",
        "environment(name = 'foo')",
        "environment(name = 'bar')",
        "environment_group(name='group', environments = [':foo'], defaults = [':bar'])");
  }

  @Test
  public void testEnvironmentGroupDuplicateEnvironments() throws Exception {
    expectEvalError(
        "label '//pkg:foo' is duplicated in the 'environments' list of 'group'",
        "environment(name = 'foo')",
        "environment_group(name='group', environments = [':foo', ':foo'], defaults = [':foo'])");
  }

  @Test
  public void testEnvironmentGroupDuplicateDefaults() throws Exception {
    expectEvalError(
        "label '//pkg:foo' is duplicated in the 'defaults' list of 'group'",
        "environment(name = 'foo')",
        "environment_group(name='group', environments = [':foo'], defaults = [':foo', ':foo'])");
  }

  @Test
  public void testMultipleEnvironmentGroupsValidMembership() throws Exception {
    expectEvalSuccess(
        "environment(name = 'foo')",
        "environment(name = 'bar')",
        "environment_group(name='foo_group', environments = [':foo'], defaults = [':foo'])",
        "environment_group(name='bar_group', environments = [':bar'], defaults = [':bar'])");
  }

  @Test
  public void testMultipleEnvironmentGroupsConflictingMembership() throws Exception {
    expectEvalError(
        "environment //pkg:foo belongs to both //pkg:bar_group and //pkg:foo_group",
        "environment(name = 'foo')",
        "environment(name = 'bar')",
        "environment_group(name='foo_group', environments = [':foo'], defaults = [':foo'])",
        "environment_group(name='bar_group', environments = [':foo'], defaults = [':foo'])");
  }

  @Test
  public void testFulfillsReferencesWrongTargetType() throws Exception {
    expectEvalError(
        "in \"fulfills\" attribute of //pkg:foo: //pkg:bar is not a valid environment",
        "environment(name = 'foo', fulfills = [':bar'])",
        "cc_library(name = 'bar')",
        "environment_group(name='foo_group', environments = [':foo'], defaults = [])");
  }

  @Test
  public void testFulfillsNotInEnvironmentGroup() throws Exception {
    expectEvalError(
        "in \"fulfills\" attribute of //pkg:foo: //pkg:bar is not a member of this group",
        "environment(name = 'foo', fulfills = [':bar'])",
        "environment(name = 'bar')",
        "environment_group(name='foo_group', environments = [':foo'], defaults = [])");
  }

  @Test
  public void testPackageDefaultEnvironments() throws Exception {
    Package pkg =
        expectEvalSuccess(
            "package(",
            "    default_compatible_with=['//foo'],",
            "    default_restricted_to=['//bar'],",
            ")");
    assertThat(pkg.getDefaultCompatibleWith())
        .containsExactly(Label.parseAbsolute("//foo", ImmutableMap.of()));
    assertThat(pkg.getDefaultRestrictedTo())
        .containsExactly(Label.parseAbsolute("//bar", ImmutableMap.of()));
  }

  @Test
  public void testPackageDefaultCompatibilityDuplicates() throws Exception {
    expectEvalError(
        "'//foo:foo' is duplicated in the 'default_compatible_with' list",
        "package(default_compatible_with=['//foo', '//bar', '//foo'])");
  }

  @Test
  public void testPackageDefaultRestrictionDuplicates() throws Exception {
    expectEvalError(
        "'//foo:foo' is duplicated in the 'default_restricted_to' list",
        "package(default_restricted_to=['//foo', '//bar', '//foo'])");
  }

  @Test
  public void testGlobPatternExtractor() {
    StarlarkFile file =
        StarlarkFile.parse(
            ParserInput.fromLines(
                "pattern = '*'",
                "some_variable = glob([",
                "  '**/*',",
                "  'a' + 'b',",
                "  pattern,",
                "])",
                "other_variable = glob(include = ['a'], exclude = ['b'])",
                "third_variable = glob(['c'], exclude_directories = 0)"));
    List<String> globs = new ArrayList<>();
    List<String> globsWithDirs = new ArrayList<>();
    PackageFactory.checkBuildSyntax(
        file, globs, globsWithDirs, new HashMap<>(), /*eventHandler=*/ null);
    assertThat(globs).containsExactly("ab", "a", "**/*");
    assertThat(globsWithDirs).containsExactly("c");
  }

  // Tests of BUILD file dialect checks:

  @Test
  public void testDefInBuild() throws Exception {
    checkBuildDialectError(
        "def func(): pass", //
        "function definitions are not allowed in BUILD files");
  }

  @Test
  public void testForStatementForbiddenInBuild() throws Exception {
    checkBuildDialectError(
        "for _ in []: pass", //
        "for loops are not allowed");
  }

  @Test
  public void testIfStatementForbiddenInBuild() throws Exception {
    checkBuildDialectError(
        "if False: pass", //
        "if statements are not allowed");
  }

  @Test
  public void testKwargsForbiddenInBuild() throws Exception {
    checkBuildDialectError(
        "print(**dict)", //
        "**kwargs arguments are not allowed in BUILD files");
    checkBuildDialectError(
        "len(dict(**{'a': 1}))", //
        "**kwargs arguments are not allowed in BUILD files");
  }

  @Test
  public void testArgsForbiddenInBuild() throws Exception {
    checkBuildDialectError(
        "print(*['a'])", //
        "*args arguments are not allowed in BUILD files");
  }

  // Asserts that evaluation of the specified BUILD file produces the expected error.
  // Modifies: scratch, events, packages; be careful when calling more than once per @Test!
  private void checkBuildDialectError(String content, String expectedError)
      throws IOException, InterruptedException, NoSuchPackageException {
    events.clear();
    events.setFailFast(false);
    Path file = scratch.overwriteFile("/p/BUILD", content);
    Package pkg = packages.eval("p", RootedPath.toRootedPath(root, file));
    assertThat(pkg.containsErrors()).isTrue();
    events.assertContainsError(expectedError);
  }
}
