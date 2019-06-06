// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static com.google.common.truth.Fact.simpleFact;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Testing Subject for comparing ParsedAndroidData instances. */
class ParsedAndroidDataSubject extends Subject {

  private final ParsedAndroidData actual;

  public ParsedAndroidDataSubject(FailureMetadata failureMetadata, ParsedAndroidData actual) {
    super(failureMetadata, actual);
    this.actual = actual;
  }

  public void isEqualTo(ParsedAndroidData expectation) {
    List<String> errors = new ArrayList<>();
    this.<DataAsset>compareDataValues(
        actual.iterateAssetEntries(), expectation.iterateAssetEntries(), errors, "assets");
    this.<DataResource>compareDataValues(
        actual.iterateCombiningEntries(),
        expectation.iterateCombiningEntries(),
        errors,
        "combining");
    this.<DataResource>compareDataValues(
        actual.iterateOverwritableEntries(),
        expectation.iterateOverwritableEntries(),
        errors,
        "overwritable");
    if (!errors.isEmpty()) {
      failWithoutActual(simpleFact(Joiner.on("\n").join(errors)));
    }
  }

  private <T extends DataValue> void compareDataValues(
      Iterable<Map.Entry<DataKey, T>> actual,
      Iterable<Map.Entry<DataKey, T>> expected,
      List<String> out,
      String valueType) {
    List<String> errors = new ArrayList<>();
    ImmutableMap<DataKey, T> actualMap = ImmutableMap.copyOf(actual);
    ImmutableMap<DataKey, T> expectedMap = ImmutableMap.copyOf(expected);
    for (DataKey key : Sets.union(actualMap.keySet(), expectedMap.keySet())) {
      if (!(actualMap.containsKey(key) && expectedMap.containsKey(key))) {
        if (!actualMap.containsKey(key)) {
          errors.add(error("\tExpected %s.", key.toPrettyString()));
        }
        if (!expectedMap.containsKey(key)) {
          errors.add(error("\tHad unexpected %s.", key.toPrettyString()));
        }
      } else {
        T actualValue = actualMap.get(key);
        T expectedValue = expectedMap.get(key);
        if (!actualValue.equals(expectedValue)) {
          errors.add(error("\t%s is not equal", key.toPrettyString()));
          if (!actualValue.source().equals(expectedValue.source())) {
            if (!actualValue.source().getPath().equals(expectedValue.source().getPath())) {
              errors.add(error("\t\t%-10s: %s", "Expected path", expectedValue.source().getPath()));
              errors.add(error("\t\t%-10s: %s", "Actual path", actualValue.source().getPath()));
            }
            if (!actualValue.source().overrides().equals(expectedValue.source().overrides())) {
              errors.add(error("\t\t%-10s: %s", "Expected overrides", expectedValue.source()));
              errors.add(error("\t\t%-10s: %s", "Actual overrides", actualValue.source()));
            }
          }
          if (!actualValue.getClass().equals(expectedValue.getClass())) {
            errors.add(error("\t\t%-10s: %s", "Expected class", expectedValue.getClass()));
            errors.add(error("\t\t%-10s: %s", "Actual class", actualValue.getClass()));
          } else if (actualValue instanceof DataResourceXml) {
            errors.add(error("\t\t%-10s: %s", "Expected xml", expectedValue));
            errors.add(error("\t\t%-10s: %s", "Actual xml", actualValue));
          }
        }
      }
    }
    if (errors.isEmpty()) {
      return;
    }
    out.add(valueType);
    out.add(Joiner.on("\n").join(errors));
  }

  String error(String template, Object... args) {
    return String.format(template, args);
  }
}
