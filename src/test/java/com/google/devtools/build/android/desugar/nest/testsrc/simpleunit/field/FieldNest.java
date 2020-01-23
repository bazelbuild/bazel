// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.field;

/** A nest for testing private field desugaring. */
public class FieldNest {

  /** A nest member that encloses private fields with cross-mate reads and writes. */
  public static class FieldOwnerMate {

    static long staticField = 10L;

    int instanceField = 20;

    private static long privateStaticField = 30L;

    private int privateInstanceField = 40;

    private long privateInstanceWideField = 45L;

    private static long privateStaticFieldReadOnly = 50L;

    private int privateInstanceFieldReadOnly = 60;

    private static long privateStaticFieldInBoundary = 70L;

    private int privateInstanceFieldInBoundary = 80;

    private static long[] privateStaticArrayField = {90L, 100L};

    private int[] privateInstanceArrayField = {110, 120};

    public static long getPrivateStaticFieldInBoundary() {
      return privateStaticFieldInBoundary;
    }

    public int getPrivateInstanceFieldInBoundary() {
      return privateInstanceFieldInBoundary;
    }
  }

  public static synchronized long getStaticField() {
    return FieldOwnerMate.staticField;
  }

  public static synchronized int getInstanceField(FieldOwnerMate mate) {
    return mate.instanceField;
  }

  public static synchronized long getPrivateStaticField() {
    return FieldOwnerMate.privateStaticField;
  }

  public static synchronized long setPrivateStaticField(long x) {
    return FieldOwnerMate.privateStaticField = x;
  }

  public static synchronized int getPrivateInstanceField(FieldOwnerMate mate) {
    return mate.privateInstanceField;
  }

  public static synchronized int setPrivateInstanceField(FieldOwnerMate mate, int x) {
    return mate.privateInstanceField = x;
  }

  public static synchronized long setPrivateInstanceWideField(FieldOwnerMate mate, long x) {
    return mate.privateInstanceWideField = x;
  }

  public static synchronized long getPrivateStaticArrayFieldElement(int index) {
    return FieldOwnerMate.privateStaticArrayField[index];
  }

  public static synchronized long setPrivateStaticArrayFieldElement(int index, long value) {
    return FieldOwnerMate.privateStaticArrayField[index] = value;
  }

  public static synchronized int getPrivateInstanceArrayFieldElement(
      FieldOwnerMate mate, int index) {
    return mate.privateInstanceArrayField[index];
  }

  public static synchronized int setPrivateInstanceArrayFieldElement(
      FieldOwnerMate mate, int index, int value) {
    return mate.privateInstanceArrayField[index] = value;
  }

  public static synchronized long getPrivateStaticFieldReadOnly() {
    return FieldOwnerMate.privateStaticFieldReadOnly;
  }

  public static synchronized int getPrivateInstanceFieldReadOnly(FieldOwnerMate mate) {
    return mate.privateInstanceFieldReadOnly;
  }

  public static synchronized long getPrivateStaticFieldInBoundary() {
    return FieldOwnerMate.getPrivateStaticFieldInBoundary();
  }

  public static synchronized int getPrivateInstanceFieldInBoundary(FieldOwnerMate mate) {
    return mate.getPrivateInstanceFieldInBoundary();
  }

  public static synchronized long compoundSetPrivateStaticField(long x) {
    return FieldOwnerMate.privateStaticField += x;
  }

  public static synchronized long preIncrementPrivateStaticField() {
    return ++FieldOwnerMate.privateStaticField;
  }

  public static synchronized long postIncrementPrivateStaticField() {
    return FieldOwnerMate.privateStaticField++;
  }

  public static synchronized int compoundSetPrivateInstanceField(FieldOwnerMate mate, int x) {
    return mate.privateInstanceField += x;
  }

  public static synchronized int preIncrementPrivateInstanceField(FieldOwnerMate mate) {
    return ++mate.privateInstanceField;
  }

  public static synchronized int postIncrementPrivateInstanceField(FieldOwnerMate mate) {
    return mate.privateInstanceField++;
  }
}
