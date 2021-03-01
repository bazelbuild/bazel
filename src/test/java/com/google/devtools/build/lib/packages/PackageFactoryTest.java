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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.PackageValidator.InvalidPackageException;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import net.starlark.java.eval.Starlark;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.StarlarkFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@code PackageFactory}. Note: PackageLoadingTestCase doesn't support REPOSITORY
 * skyframe function, thus these tests cannot load external packages, {@code @repo://pkg}.
 */
@RunWith(JUnit4.class)
public final class PackageFactoryTest extends PackageLoadingTestCase {

  private Path throwOnReaddir = null;

  // Overrides FileSystem.readdir for the benefit of one test method
  // (testTransientErrorsInGlobbing) that injects a failure.
  @Override
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(DigestHashFunction.SHA256) {
      @Override
      public Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
        if (path.equals(throwOnReaddir)) {
          throw new FileNotFoundException(path.getPathString());
        }
        return super.readdir(path, followSymlinks);
      }
    };
  }

  @Test
  public void testCreatePackage() throws Exception {
    scratch.file("pkgname/BUILD", "# empty build file ");
    Package pkg = loadPackage("pkgname");
    assertThat(pkg.getName()).isEqualTo("pkgname");
    assertThat(Sets.newHashSet(pkg.getTargets(Rule.class))).isEmpty();
  }

  @Test
  public void testBadRuleName() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("badrulename/BUILD", "cc_library(name = 3)");
    Package pkg = loadPackage("badrulename");
    assertContainsEvent("cc_library 'name' attribute must be a string");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testNoRuleName() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("badrulename/BUILD", "cc_library()");
    Package pkg = loadPackage("badrulename");
    assertContainsEvent("cc_library rule has no 'name' attribute");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testBadPackageName() throws Exception {
    // This is a "shallow" syntactic error: failure to form the
    // PackageIdentifier that is the real argument to loadPackage.
    LabelSyntaxException e =
        assertThrows(LabelSyntaxException.class, () -> loadPackage("not even a legal/.../label"));
    assertThat(e).hasMessageThat().contains("invalid package name 'not even a legal/.../label'");
  }

  @Test
  public void testColonInExportsFilesTargetName() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "googledata/cafe/BUILD", "exports_files(['houseads/house_ads:ca-aol_parenting_html'])");
    Package pkg = loadPackage("googledata/cafe");
    assertContainsEvent("target names may not contain ':'");
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
            + " 'sequence or NoneType'",
        "exports_files(srcs=[], licenses=depset(['notice']))");
  }

  @Test
  public void testPackageNameWithPROTECTEDIsOk() throws Exception {
    reporter.removeHandler(failFastHandler);
    // One "PROTECTED":
    assertThat(isValidPackageName("foo/PROTECTED/bar")).isTrue();
    // Multiple "PROTECTED"s:
    assertThat(isValidPackageName("foo/PROTECTED/bar/PROTECTED/wiz")).isTrue();
  }

  private boolean isValidPackageName(String packageName) throws Exception {
    // Write a license decl just in case it's a third_party package:
    scratch.file(packageName + "/BUILD", "licenses(['notice'])");
    Package pkg = loadPackage(packageName);
    return !pkg.containsErrors();
  }

  @Test
  public void testDuplicatedDependencies() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "has_dupe/BUILD",
        "cc_library(name='dep')",
        "cc_library(name='has_dupe', deps=[':dep', ':dep'])");

    Package pkg = loadPackage("has_dupe");
    assertContainsEvent(
        "Label '//has_dupe:dep' is duplicated in the 'deps' attribute of rule 'has_dupe'");
    assertThat(pkg.containsErrors()).isTrue();
    assertThat(pkg.getRule("has_dupe")).isNotNull();
    assertThat(pkg.getRule("dep")).isNotNull();
    assertThat(pkg.getRule("has_dupe").containsErrors()).isTrue();
    // All rules in an errant package are themselves errant.
    assertThat(pkg.getRule("dep").containsErrors()).isTrue();
  }

  @Test
  public void testPrefixWithinSameRule1() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "fruit/orange/BUILD", "genrule(name='orange', srcs=[], outs=['a', 'a/b'], cmd='')");
    loadPackage("fruit/orange");
    assertContainsEvent("rule 'orange' has conflicting output files 'a/b' and 'a");
  }

  @Test
  public void testPrefixWithinSameRule2() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "fruit/orange/BUILD", "genrule(name='orange', srcs=[], outs=['a/b', 'a'], cmd='')");
    loadPackage("fruit/orange");
    assertContainsEvent("rule 'orange' has conflicting output files 'a' and 'a/b");
  }

  @Test
  public void testPrefixBetweenRules1() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "fruit/kiwi/BUILD",
        "genrule(name='kiwi1', srcs=[], outs=['a'], cmd='')",
        "genrule(name='kiwi2', srcs=[], outs=['a/b'], cmd='')");
    loadPackage("fruit/kiwi");
    assertContainsEvent(
        "output file 'a/b' of rule 'kiwi2' conflicts with output file 'a' of rule 'kiwi1'");
  }

  @Test
  public void testPrefixBetweenRules2() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "fruit/kiwi/BUILD",
        "genrule(name='kiwi1', srcs=[], outs=['a/b'], cmd='')",
        "genrule(name='kiwi2', srcs=[], outs=['a'], cmd='')");
    loadPackage("fruit/kiwi");
    assertContainsEvent(
        "output file 'a' of rule 'kiwi2' conflicts with output file 'a/b' of rule 'kiwi1'");
  }

  @Test
  public void testPackageNameFunction() throws Exception {
    scratch.file("pina/BUILD", "cc_library(name=package_name() + '-colada')");
    Package pkg = loadPackage("pina");
    assertNoEvents();
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getRule("pina-colada")).isNotNull();
    assertThat(pkg.getRule("pina-colada").containsErrors()).isFalse();
    assertThat(Sets.newHashSet(pkg.getTargets(Rule.class)).size()).isSameInstanceAs(1);
  }

  @Test
  public void testDuplicateRuleName() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "duplicaterulename/BUILD",
        "proto_library(name = 'spellcheck_proto',",
        "         srcs = ['spellcheck.proto'],",
        "         cc_api_version = 2)",
        "cc_library(name = 'spellcheck_proto')", // conflict error stops execution
        "x = 1//0"); // not reached
    Package pkg = loadPackage("duplicaterulename");
    assertContainsEvent(
        "cc_library rule 'spellcheck_proto' in package 'duplicaterulename' conflicts with"
            + " existing proto_library rule");
    assertDoesNotContainEvent("division by zero");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testBuildFileTargetExists() throws Exception {
    scratch.file("foo/BUILD");
    Target target = getTarget("//foo:BUILD");
    assertThat(target.getName()).isEqualTo("BUILD");
    // Test that it's memoized:
    assertThat(target.getPackage().getTarget("BUILD")).isSameInstanceAs(target);
  }

  @Test
  public void testCreationOfInputFiles() throws Exception {
    scratch.file(
        "foo/BUILD",
        "exports_files(['Z'])",
        "cc_library(name='W', deps=['X', 'Y'])",
        "cc_library(name='X', srcs=['X'])",
        "cc_library(name='Y')");
    Package pkg = loadPackage("foo");
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
                + "target 'A' not declared in package 'foo' defined by /workspace/foo/BUILD");

    // These are the only input files: BUILD, Z
    Set<String> inputFiles = Sets.newTreeSet();
    for (InputFile inputFile : pkg.getTargets(InputFile.class)) {
      inputFiles.add(inputFile.getName());
    }
    assertThat(Lists.newArrayList(inputFiles)).containsExactly("BUILD", "Z").inOrder();
  }

  @Test
  public void testDuplicateRuleIsNotAddedToPackage() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "dup/BUILD",
        "proto_library(name = 'dup_proto',",
        "              srcs  = ['dup.proto'],",
        "              cc_api_version = 2)",
        "",
        "cc_library(name = 'dup_proto',",
        "           srcs = ['dup.pb.cc', 'dup.pb.h'])");
    Package pkg = loadPackage("dup");
    assertContainsEvent(
        "cc_library rule 'dup_proto' in package 'dup' conflicts with existing proto_library rule");
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
    reporter.removeHandler(failFastHandler);

    // In this test, rule2's outputs conflict with rule1, so rule2 is rejected.
    // However, we must check that neither rule2, nor any of its inputs or
    // outputs is a member of the package, and that the conflicting output file
    // "out2" still has rule1 as its getGeneratingRule().
    scratch.file(
        "conflict/BUILD",
        "genrule(name = 'rule1',",
        "        cmd = '',",
        "        srcs = ['in1', 'in2'],",
        "        outs = ['out1', 'out2'])",
        "genrule(name = 'rule2',",
        "        cmd = '',",
        "        srcs = ['in3', 'in4'],",
        "        outs = ['out3', 'out2'])");
    Package pkg = loadPackage("conflict");
    assertContainsEvent(
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
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "error/BUILD",
        "genrule(name = 'rule1',",
        "        cmd = ':',",
        "        outs = ['out.1'])",
        "list = ['bad']",
        "x = 1//0", // dynamic error
        "genrule(name = 'rule2',",
        "        cmd = ':',",
        "        outs = list)");
    Package pkg = loadPackage("error");
    assertContainsEvent("division by zero");

    assertThat(pkg.containsErrors()).isTrue();

    // rule1 would be fine but is still marked as in error:
    assertThat(pkg.getRule("rule1").containsErrors()).isTrue();

    // rule2's genrule is never executed.
    Rule rule2 = pkg.getRule("rule2");
    assertThat(rule2).isNull();
  }

  @Test
  public void testHelpfulErrorForMissingExportsFiles() throws Exception {
    scratch.file("x/BUILD", "cc_library(name='x', srcs=['x.cc'])");
    scratch.file("x/x.cc");
    scratch.file("x/y.cc");
    scratch.file("x/dir/dummy");

    Package pkg = loadPackage("x");

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
                + "defined by /workspace/x/BUILD");

    e = assertThrows(NoSuchTargetException.class, () -> pkg.getTarget("z.cc"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "no such target '//x:z.cc': "
                + "target 'z.cc' not declared in package 'x' (did you mean 'x.cc'?) "
                + "defined by /workspace/x/BUILD");

    e = assertThrows(NoSuchTargetException.class, () -> pkg.getTarget("dir"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "no such target '//x:dir': target 'dir' not declared in package 'x'; "
                + "however, a source directory of this name exists.  "
                + "(Perhaps add 'exports_files([\"dir\"])' to x/BUILD, "
                + "or define a filegroup?) defined by /workspace/x/BUILD");
  }

  @Test
  public void testTestSuitesImplicitlyDependOnAllRulesInPackage() throws Exception {
    scratch.file(
        "x/BUILD",
        "java_test(name='j')",
        "test_suite(name='t1')",
        "test_suite(name='t2', tests=[])",
        "test_suite(name='t3', tests=['//foo'])",
        "test_suite(name='t4', tests=['//foo'])",
        "cc_test(name='c')");
    Package pkg = loadPackage("x");

    // Things to note:
    // - The '$implicit_tests' attribute is unset unless the 'tests' attribute is unset or empty.
    // - The '$implicit_tests' attribute's value for t1 and t2 is magically able to contain both j
    //    and c, even though c is instantiated after t1 and t2 are.

    assertThat(attributes(pkg.getRule("t1")).get("$implicit_tests", BuildType.LABEL_LIST))
        .containsExactlyElementsIn(
            Sets.newHashSet(
                Label.parseAbsolute("//x:c", ImmutableMap.of()),
                Label.parseAbsolute("//x:j", ImmutableMap.of())));
    assertThat(attributes(pkg.getRule("t2")).get("$implicit_tests", BuildType.LABEL_LIST))
        .containsExactlyElementsIn(
            Sets.newHashSet(
                Label.parseAbsolute("//x:c", ImmutableMap.of()),
                Label.parseAbsolute("//x:j", ImmutableMap.of())));
    assertThat(attributes(pkg.getRule("t3")).get("$implicit_tests", BuildType.LABEL_LIST))
        .isEmpty();
    assertThat(attributes(pkg.getRule("t4")).get("$implicit_tests", BuildType.LABEL_LIST))
        .isEmpty();
  }

  @Test
  public void testPackageValidationFailureRegisteredAfterLoading() throws Exception {
    scratch.file("x/BUILD", "# old");
    Package pkg = loadPackage("x");
    assertThat(pkg.containsErrors()).isFalse();

    // Install a validator.
    this.validator =
        (pkg2, eventHandler) -> {
          if (pkg2.getName().equals("x")) {
            eventHandler.handle(Event.warn("warning event"));
            throw new InvalidPackageException(pkg2.getPackageIdentifier(), "nope");
          }
        };

    scratch.overwriteFile("x/BUILD", "# new"); // change file to cause reloading
    invalidatePackages();

    InvalidPackageException ex =
        assertThrows(InvalidPackageException.class, () -> loadPackage("x"));
    assertThat(ex).hasMessageThat().contains("no such package 'x': nope");
    assertContainsEvent("warning event");
  }

  @Test
  public void testGlobDirectoryExclusion() throws Exception {
    emptyFile("fruit/data/apple");
    emptyFile("fruit/data/pear");
    emptyFile("fruit/data/berry/black");
    emptyFile("fruit/data/berry/blue");
    scratch.file(
        "fruit/BUILD",
        "cc_library(name = 'yes', srcs = glob(['data/*']))",
        "cc_library(name = 'no',  srcs = glob(['data/*'], exclude_directories=0))");
    Package pkg = loadPackage("fruit");
    assertNoEvents();
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
    emptyFile("rg/a.cc");
    emptyFile("rg/foo/bar.cc");
    emptyFile("rg/foo/foo.cc");
    emptyFile("rg/foo/wiz/bam.cc");
    emptyFile("rg/foo/wiz/bum.cc");
    emptyFile("rg/foo/wiz/quid/gav.cc");
    scratch.file(
        "rg/BUILD",
        "cc_library(name = 'ri', srcs = glob(['**/*.cc']))",
        "cc_library(name = 're', srcs = glob(['*.cc'], exclude=['**/*.c']))");
    Package pkg = loadPackage("rg");
    assertNoEvents();

    assertGlob(
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

    assertGlob(
        pkg,
        ImmutableList.of(
            "a.cc",
            "foo/bar.cc",
            "foo/foo.cc",
            "foo/wiz/bam.cc",
            "foo/wiz/bum.cc",
            "foo/wiz/quid/gav.cc"),
        "**/*.cc");
    assertGlob(pkg, ImmutableList.of("foo/bar.cc", "foo/wiz/bam.cc", "foo/wiz/bum.cc"), "**/b*.cc");
    assertGlob(
        pkg,
        ImmutableList.of(
            "foo/bar.cc", "foo/foo.cc", "foo/wiz/bam.cc", "foo/wiz/bum.cc", "foo/wiz/quid/gav.cc"),
        "**/*/*.cc");
    assertGlob(pkg, ImmutableList.of("foo/wiz/quid/gav.cc"), "foo/**/quid/*.cc");

    assertGlob(
        pkg,
        Collections.<String>emptyList(),
        ImmutableList.of("*.cc", "*/*.cc", "*/*/*.cc"),
        ImmutableList.of("**/*.cc"));
    assertGlob(
        pkg,
        Collections.<String>emptyList(),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("**/*.cc"));
    assertGlob(
        pkg,
        Collections.<String>emptyList(),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("*.cc", "*/*.cc", "*/*/*.cc", "*/*/*/*.cc"));
    assertGlob(
        pkg,
        Collections.<String>emptyList(),
        ImmutableList.of("**"),
        ImmutableList.of("*", "*/*", "*/*/*", "*/*/*/*"));
    assertGlob(
        pkg,
        ImmutableList.of(
            "foo/bar.cc", "foo/foo.cc", "foo/wiz/bam.cc", "foo/wiz/bum.cc", "foo/wiz/quid/gav.cc"),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("*.cc"));
    assertGlob(
        pkg,
        ImmutableList.of("a.cc", "foo/wiz/bam.cc", "foo/wiz/bum.cc", "foo/wiz/quid/gav.cc"),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("*/*.cc"));
    assertGlob(
        pkg,
        ImmutableList.of("a.cc", "foo/bar.cc", "foo/foo.cc", "foo/wiz/quid/gav.cc"),
        ImmutableList.of("**/*.cc"),
        ImmutableList.of("**/wiz/*.cc"));
  }

  @Test
  public void testTooManyArgumentsGlobErrors() throws Exception {
    reporter.removeHandler(failFastHandler);
    assertGlobFails(
        "glob(['incl'],['excl'],3,True,'extraarg')",
        "glob() accepts no more than 4 positional arguments but got 5");
  }

  @Test
  public void testGlobEnforcesListArgument() throws Exception {
    reporter.removeHandler(failFastHandler);
    assertGlobFails(
        "glob(1, exclude=2)",
        "in call to glob(), parameter 'include' got value of type 'int', want 'sequence'");
  }

  @Test
  public void testGlobEnforcesListOfStringsArguments() throws Exception {
    reporter.removeHandler(failFastHandler);
    assertGlobFails(
        "glob(['a', 'b'], exclude=['c', 42])",
        "expected value of type 'string' for element 1 of 'glob' argument, but got 42 (int)");
  }

  @Test
  public void testGlobNegativeTest() throws Exception {
    // Negative test that assertGlob does throw an error when asserting against the wrong values.
    // The AssertionError comes from FoundationTestCase.failFastHandler.
    AssertionError e =
        assertThrows(
            AssertionError.class,
            () ->
                assertGlobMatches(
                    /*result=*/ ImmutableList.of("Wombat1.java", "This_file_doesn_t_exist.java"),
                    /*includes=*/ ImmutableList.of("W*", "subdir"),
                    /*excludes=*/ ImmutableList.<String>of(),
                    /* excludeDirs= */ true));
    assertThat(e).hasMessageThat().contains("incorrect glob result");
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
    reporter.removeHandler(failFastHandler);
    assertGlobFails("glob(['?'])", "Error in glob: wildcard ? forbidden");
  }

  @Test
  public void testBadExcludePattern() throws Exception {
    reporter.removeHandler(failFastHandler);
    // The 'exclude' check is currently only reached if the pattern is "complex".
    // This seems like a bug:
    //   assertGlobFails("glob(['BUILD'], ['/'])", "pattern cannot be absolute");
    assertGlobFails("glob(['BUILD'], ['/*/*'])", "pattern cannot be absolute");
  }

  @Test
  public void testGlobEscapesAt() throws Exception {
    // See lib.skyframe.PackageFunctionTest.globEscapesAt and
    // https://github.com/bazelbuild/bazel/issues/10606.
    scratch.file("p/@f.txt");
    scratch.file(
        "p/BUILD",
        "name = glob(['*.txt'])[0]",
        "name == ':@f.txt' or fail('got %s' % name)"); // observe prepended colon
    loadPackage("p"); // no error
  }

  /**
   * Tests that a glob evaluation that encounters an I/O error throws instead of constructing a
   * package.
   */
  @Test
  public void testGlobWithIOErrors() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("pkg/BUILD", "glob(['globs/**'])");
    Path dir = scratch.dir("pkg/globs/unreadable");
    dir.setReadable(false);

    NoSuchPackageException ex =
        assertThrows(NoSuchPackageException.class, () -> loadPackage("pkg"));
    assertThat(ex)
        .hasMessageThat()
        .contains("error globbing [globs/**]: " + dir + " (Permission denied)");
  }

  @Test
  public void testNativeModuleIsDisabled() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("pkg/BUILD", "native.cc_library(name='bar')");
    Package pkg = loadPackage("pkg");
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
    scratch.file(
        "foo/BUILD",
        "package(default_deprecation = \"" + msg + "\")",
        "sh_library(name = 'bar', srcs=['b'])");
    Rule fooRule = (Rule) getTarget("//foo:bar");
    String deprAttr =
        attributes(fooRule).get("deprecation", com.google.devtools.build.lib.packages.Type.STRING);
    assertThat(deprAttr).isEqualTo(msg);
  }

  @Test
  public void testDefaultTestonlyPropagation() throws Exception {
    scratch.file(
        "foo/BUILD",
        "package(default_testonly = 1)",
        "sh_library(name = 'foo', srcs=['b'])",
        "sh_library(name = 'bar', srcs=['b'], testonly = 0)");
    Package pkg = loadPackage("foo");

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
    scratch.file(
        "foo/BUILD",
        "package(default_deprecation = \"" + deceive + "\")",
        "sh_library(name = 'bar', srcs=['b'], deprecation = \"" + msg + "\")");
    Package pkg = loadPackage("foo");

    Rule fooRule = (Rule) pkg.getTarget("bar");
    String deprAttr =
        attributes(fooRule).get("deprecation", com.google.devtools.build.lib.packages.Type.STRING);
    assertThat(deprAttr).isEqualTo(msg);
  }

  @Test
  public void testPackageFeatures() throws Exception {
    scratch.file(
        "a/BUILD",
        "sh_library(name='before')",
        "package(features=['b', 'c'])",
        "sh_library(name='after')");
    Package pkg = loadPackage("a");
    assertThat(pkg.getFeatures()).containsExactly("b", "c");
  }

  @Test
  public void testTransientErrorsInGlobbing() throws Exception {
    Path buildFile = scratch.file("e/BUILD", "sh_library(name = 'e', data = glob(['*']))");
    throwOnReaddir = buildFile.getParentDirectory();
    invalidatePackages();
    reporter.removeHandler(failFastHandler);
    assertThrows(NoSuchPackageException.class, () -> loadPackage("e")); // symlink cycle

    throwOnReaddir = null;
    invalidatePackages();

    reporter.addHandler(failFastHandler);
    Package pkg = loadPackage("e"); // no error
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getRule("e")).isNotNull();
    List<?> globList = (List) pkg.getRule("e").getAttr("data");
    assertThat(globList).containsExactly(Label.parseAbsolute("//e:BUILD", ImmutableMap.of()));
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
        "functions may not be defined in BUILD files");
  }

  @Test
  public void testLambdaInBuild() throws Exception {
    checkBuildDialectError(
        "lambda: None", //
        "functions may not be defined in BUILD files");
  }

  @Test
  public void testForStatementForbiddenInBuild() throws Exception {
    checkBuildDialectError(
        "for _ in []: pass", //
        "for statements are not allowed in BUILD files");
  }

  @Test
  public void testIfStatementForbiddenInBuild() throws Exception {
    checkBuildDialectError(
        "if False: pass", //
        "if statements are not allowed in BUILD files");
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
  private void checkBuildDialectError(String content, String expectedError) throws Exception {
    eventCollector.clear();
    reporter.removeHandler(failFastHandler);
    scratch.overwriteFile("p/BUILD", content);
    invalidatePackages();
    Package pkg = loadPackage("p");
    assertContainsEvent(expectedError);
    assertThat(pkg.containsErrors()).isTrue();
  }

  private Package expectEvalSuccess(String... content) throws Exception {
    scratch.file("pkg/BUILD", content);
    Package pkg = loadPackage("pkg");
    assertThat(pkg.containsErrors()).isFalse();
    return pkg;
  }

  private void expectEvalError(String expectedError, String... content) throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("pkg/BUILD", content);
    Package pkg = loadPackage("pkg");
    assertWithMessage("Expected evaluation error, but none was not reported")
        .that(pkg.containsErrors())
        .isTrue();
    assertContainsEvent(expectedError);
  }

  private static AttributeMap attributes(Rule rule) {
    return RawAttributeMapper.of(rule);
  }

  private static void assertGlob(Package pkg, List<String> expected, String... include)
      throws Exception {
    assertGlob(pkg, expected, ImmutableList.copyOf(include), ImmutableList.of());
  }

  private static void assertGlob(
      Package pkg, List<String> expected, List<String> include, List<String> exclude)
      throws Exception {
    GlobCache globCache =
        new GlobCache(
            pkg.getFilename().asPath().getParentDirectory(),
            pkg.getPackageIdentifier(),
            ImmutableSet.of(),
            // a package locator that finds no packages
            new CachingPackageLocator() {
              @Override
              public Path getBuildFileForPackage(PackageIdentifier packageName) {
                return null;
              }
            },
            null,
            TestUtils.getPool(),
            -1);
    assertThat(globCache.globUnsorted(include, exclude, false, true))
        .containsExactlyElementsIn(expected);
  }

  private Path emptyFile(String path) {
    try {
      return scratch.file(path);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  /********************************************************************
   *                                                                  *
   *              Test "glob" function in build language              *
   *                                                                  *
   ********************************************************************/
  private void assertGlobFails(String globCallExpression, String expectedError) throws Exception {
    Package pkg = buildPackageWithGlob(globCallExpression);

    assertContainsEvent(expectedError);
    assertThat(pkg.containsErrors()).isTrue();
  }

  private Package buildPackageWithGlob(String globCallExpression) throws Exception {
    scratch.deleteFile("dummypackage/BUILD");
    scratch.file("dummypackage/BUILD", "x = " + globCallExpression);
    return loadPackage("dummypackage");
  }

  /**
   * Test globbing in the context of a package, using the build language. We use the specially setup
   * "globs" test package and the files beneath it.
   *
   * @param result the expected list of filenames that match the glob
   * @param includes an include pattern for the glob
   * @param excludes an exclude pattern for the glob
   * @param excludeDirs an exclude_directories flag for the glob
   * @throws Exception if the glob doesn't match the expected result.
   */
  private void assertGlobMatches(
      List<String> result, List<String> includes, List<String> excludes, boolean excludeDirs)
      throws Exception {
    // If the glob doesn't match the expected result, BUILD execution calls fail() which
    // posts an ERROR to the fail-fast handler, throwing AssertionError.
    Package pkg =
        evaluateGlob(
            includes,
            excludes,
            excludeDirs,
            Starlark.format(
                "(result == sorted(%r)) or fail('incorrect glob result: got %%s, want %%s' %%"
                    + " (result, sorted(%r)))",
                result, result));
    // Execution succeeded. Assert that there were no other errors in the package.
    assertThat(pkg.containsErrors()).isFalse();
  }

  /**
   * Evaluate a glob() call against a test directory and BUILD code to process the results.
   *
   * @param includes a list of glob patterns; glob will include these files.
   * @param excludes a list of glob patterns to exclude even if previously included.
   * @param excludeDirs true if directories should be excluded from the match.
   * @param resultAssertion code in the BUILD language that can access the variable result, to which
   *     the result of the glob will be bound, and that may contain an assertion on it.
   * @throws AssertionError if any ERROR events are reported to the fail-fast handler during
   *     execution.
   */
  // TODO(adonovan): these tests would be cleaner if they did print(glob(...)) as a side effect
  // of package loading so that the caller of loadPackage can extract and return the value,
  // for @Test methods to make assertions in the usual way.
  private Package evaluateGlob(
      List<String> includes, List<String> excludes, boolean excludeDirs, String resultAssertion)
      throws Exception {
    Path globsDir = scratch.dir("globs");
    globsDir.getChild("subdir").createDirectory();
    for (String file : ImmutableList.of("Wombat1.java", "Wombat2.java", "subdir/Wombat3.java")) {
      FileSystemUtils.createEmptyFile(globsDir.getRelative(file));
    }
    scratch.file(
        "globs/BUILD",
        Starlark.format(
            "result = glob(%r, exclude=%r, exclude_directories=%r)",
            includes, excludes, excludeDirs ? 1 : 0),
        resultAssertion);
    return loadPackage("globs");
  }

  private void assertGlobProducesError(String pattern, boolean errorExpected) throws Exception {
    reporter.removeHandler(failFastHandler);
    Package pkg = evaluateGlob(ImmutableList.of(pattern), ImmutableList.of(), false, "");
    assertThat(pkg.containsErrors()).isEqualTo(errorExpected);
    boolean foundError = false;
    for (Event event : eventCollector) {
      if (event.getMessage().contains("glob")) {
        if (!errorExpected) {
          fail("error not expected for glob pattern " + pattern + ", but got: " + event);
          return;
        }
        foundError = errorExpected;
        break;
      }
    }
    assertThat(foundError).isEqualTo(errorExpected);
  }

  private Package loadPackage(String pkgid) throws Exception {
    return getTarget("//" + pkgid + ":BUILD").getPackage();
  }
}
