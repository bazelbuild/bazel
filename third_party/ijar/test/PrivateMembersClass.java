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

/**
 * A class with many private parts. Used for testing.
 */
final class PrivateMembersClass {
  private String privateField;

  private PrivateMembersClass() {}

  public static void main() {
    privateStaticMethod();
  }

  private static void privateStaticMethod() {
    new PrivateMembersClass().privateMethod();
  }

  private void privateMethod() {
    new PrivateInnerClass().print();
    new PrivateStaticInnerClass().print();
  }

  private class PrivateInnerClass {
    void print() {
      System.out.println("private inner");
    }
  }

  private static class PrivateStaticInnerClass {
    void print() {
      System.out.println("private static inner");
    }
  }

  private interface PrivateInnerInterface {}

  private @interface PrivateInnerAnnotation {}
}
