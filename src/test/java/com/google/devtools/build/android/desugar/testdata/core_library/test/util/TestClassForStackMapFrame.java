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
package test.util;

/** Test input for b/36654936 */
public class TestClassForStackMapFrame {

  /**
   * This method caused cl/152199391 to fail due to stack map frame corruption. So it is to make
   * sure the desugared version of this class still has correct stack map frames.
   */
  public String joinIntegers(int integers) {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < integers; i++) {
      if (i > 0) {
        builder.append(",");
      }
      builder.append(i);
      builder.append('=');
      Object value = i % 2 == 0 ? "Even" : "Odd";
      if (i % 2 == 0) {
        builder.append(value);
      } else {
        builder.append(value);
      }
    }
    return builder.toString();
  }

  /**
   * This method triggers ASM bug 317785 .
   *
   * @return 20
   */
  public static int testInputForAsmBug317785() {
    Integer x = 0;
    for (int i = 0; i < 10; ++i) {
      x++;
    }
    for (int i = 0; i < 10; ++i) {
      x++;
    }
    return x;
  }
}
