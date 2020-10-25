// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.query2.query.output;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.Attribute.Discriminator;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SyntheticAttributeHashCalculator}. */
@RunWith(JUnit4.class)
public class SyntheticAttributeHashCalculatorTest extends PackageLoadingTestCase {

  @Test
  public void testComputeAttributeChangeChangesHash() throws Exception {
    scratch.file("pkg/BUILD", "genrule(name='x', cmd='touch $@', outs=['y'])");
    Rule ruleBefore = (Rule) getTarget("//pkg:x");

    scratch.overwriteFile("pkg/BUILD", "genrule(name='x', cmd='touch $@', outs=['z'])");
    invalidatePackages();
    Rule ruleAfter = (Rule) getTarget("//pkg:x");

    String hashBefore =
        SyntheticAttributeHashCalculator.compute(
            ruleBefore,
            /*serializedAttributes=*/ ImmutableMap.of(),
            /*extraDataForAttrHash=*/ "",
            DigestHashFunction.SHA256.getHashFunction());
    String hashAfter =
        SyntheticAttributeHashCalculator.compute(
            ruleAfter,
            /*serializedAttributes=*/ ImmutableMap.of(),
            /*extraDataForAttrHash=*/ "",
            DigestHashFunction.SHA256.getHashFunction());

    assertThat(hashBefore).isNotEqualTo(hashAfter);
  }

  @Test
  public void testComputeLocationDoesntChangeHash() throws Exception {
    scratch.file("pkg/BUILD", "genrule(name='x', cmd='touch $@', outs=['y'])");
    Rule ruleBefore = (Rule) getTarget("//pkg:x");

    scratch.overwriteFile(
        "pkg/BUILD",
        "genrule(name='rule_that_moves_x', cmd='touch $@', outs=['whatever'])",
        "genrule(name='x', cmd='touch $@', outs=['y'])");
    invalidatePackages();
    Rule ruleAfter = (Rule) getTarget("//pkg:x");

    String hashBefore =
        SyntheticAttributeHashCalculator.compute(
            ruleBefore,
            /*serializedAttributes=*/ ImmutableMap.of(),
            /*extraDataForAttrHash=*/ "",
            DigestHashFunction.SHA256.getHashFunction());
    String hashAfter =
        SyntheticAttributeHashCalculator.compute(
            ruleAfter,
            /*serializedAttributes=*/ ImmutableMap.of(),
            /*extraDataForAttrHash=*/ "",
            DigestHashFunction.SHA256.getHashFunction());

    assertThat(hashBefore).isEqualTo(hashAfter);
  }

  @Test
  public void testComputeSerializedAttributesUsedOverAvailable() throws Exception {
    scratch.file("pkg/BUILD", "genrule(name='x', cmd='touch $@', outs=['y'])");
    Rule rule = (Rule) getTarget("//pkg:x");

    String hashBefore =
        SyntheticAttributeHashCalculator.compute(
            rule,
            /*serializedAttributes=*/ ImmutableMap.of(),
            /*extraDataForAttrHash=*/ "",
            DigestHashFunction.SHA256.getHashFunction());

    ImmutableMap<Attribute, Build.Attribute> serializedAttributes =
        ImmutableMap.of(
            rule.getRuleClassObject().getAttributeByName("cmd"),
            Build.Attribute.newBuilder()
                .setName("dummy")
                .setType(Discriminator.STRING)
                .setStringValue("hi")
                .build());

    String hashAfter =
        SyntheticAttributeHashCalculator.compute(
            rule,
            serializedAttributes, /*extraDataForAttrHash*/
            "",
            DigestHashFunction.SHA256.getHashFunction());

    assertThat(hashBefore).isNotEqualTo(hashAfter);
  }

  @Test
  public void testComputeExtraDataChangesHash() throws Exception {
    scratch.file("pkg/BUILD", "genrule(name='x', cmd='touch $@', outs=['y'])");
    Rule rule = (Rule) getTarget("//pkg:x");

    String hashBefore =
        SyntheticAttributeHashCalculator.compute(
            rule,
            /*serializedAttributes=*/ ImmutableMap.of(),
            /*extraDataForAttrHash=*/ "",
            DigestHashFunction.SHA256.getHashFunction());

    String hashAfter =
        SyntheticAttributeHashCalculator.compute(
            rule,
            /*serializedAttributes=*/ ImmutableMap.of(), /*extraDataForAttrHash*/
            "blahblaah",
            DigestHashFunction.SHA256.getHashFunction());

    assertThat(hashBefore).isNotEqualTo(hashAfter);
  }

  @Test
  public void testComputePackageErrorStatusChangesHash() throws Exception {
    scratch.file("pkg/BUILD", "genrule(name='x', cmd='touch $@', outs=['y'])");
    Rule ruleBefore = (Rule) getTarget("//pkg:x");

    // Remove fail-fast handler, we're intentionally creating a package with errors.
    reporter.removeHandler(failFastHandler);
    scratch.overwriteFile(
        "pkg/BUILD",
        "genrule(name='x', cmd='touch $@', outs=['z'])",
        "genrule(name='missing_attributes')");
    invalidatePackages();
    Rule ruleAfter = (Rule) getTarget("//pkg:x");
    assertThat(ruleAfter.containsErrors()).isTrue();

    String hashBefore =
        SyntheticAttributeHashCalculator.compute(
            ruleBefore,
            /*serializedAttributes=*/ ImmutableMap.of(),
            /*extraDataForAttrHash=*/ "",
            DigestHashFunction.SHA256.getHashFunction());
    String hashAfter =
        SyntheticAttributeHashCalculator.compute(
            ruleAfter,
            /*serializedAttributes=*/ ImmutableMap.of(),
            /*extraDataForAttrHash=*/ "",
            DigestHashFunction.SHA256.getHashFunction());

    assertThat(hashBefore).isNotEqualTo(hashAfter);
  }
}
