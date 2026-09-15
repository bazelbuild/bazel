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

import java.util.concurrent.Callable;

/**
 * For testing purposes, an anonymous inner class that requires an
 * EnclosingMethod attribute in its class file.
 */
class EnclosingMethod {
  class Inner {
    public <T> T run(Callable<T> callable) {
      try {
        return callable.call();
      } catch (Exception ex) {
        return null;
      }
    }

    private <T> Callable<T> asCallable(final Callable<T> callableToWrap) {
      return new Callable<T>() {
        @Override public T call() throws Exception {
          return Inner.this.run(callableToWrap);
        }
      };
    }
  }
}