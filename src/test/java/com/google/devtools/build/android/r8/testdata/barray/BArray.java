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
package com.google.devtools.build.android.r8.testdata.barray;

/** Test class */
@SuppressWarnings("PrivateConstructorForUtilityClass")
public class BArray {

  public static void main(String[] args) {
    System.out.println("null boolean: " + readNullBooleanArray());
    System.out.println("null byte: " + readNullByteArray());
    System.out.println("boolean: " + readBooleanArray(writeBooleanArray(args)));
    System.out.println("byte: " + readByteArray(writeByteArray(args)));
  }

  public static boolean readNullBooleanArray() {
    boolean[] boolArray = null;
    try {
      return boolArray[0] || boolArray[1];
    } catch (Throwable e) {
      return true;
    }
  }

  public static byte readNullByteArray() {
    byte[] byteArray = null;
    try {
      return byteArray[0];
    } catch (Throwable e) {
      return 42;
    }
  }

  public static boolean[] writeBooleanArray(String[] args) {
    boolean[] array = new boolean[args.length];
    for (int i = 0; i < args.length; i++) {
      array[i] = args[i].length() == 42;
    }
    return array;
  }

  public static byte[] writeByteArray(String[] args) {
    byte[] array = new byte[args.length];
    for (int i = 0; i < args.length; i++) {
      array[i] = (byte) args[i].length();
    }
    return array;
  }

  public static boolean readBooleanArray(boolean[] boolArray) {
    try {
      return boolArray[0] || boolArray[1];
    } catch (Throwable e) {
      return true;
    }
  }

  public static byte readByteArray(byte[] byteArray) {
    try {
      return byteArray[0];
    } catch (Throwable e) {
      return 42;
    }
  }
}
