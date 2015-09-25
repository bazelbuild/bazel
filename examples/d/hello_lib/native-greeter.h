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

#ifndef EXAMPLES_D_HELLO_LIB_NATIVE_GREETER_H_
#define EXAMPLES_D_HELLO_LIB_NATIVE_GREETER_H_

typedef struct NativeGreeter {
  char* greeting;
} NativeGreeter;

NativeGreeter* native_greeter_new(const char* greeting);

void native_greeter_greet(const NativeGreeter* greeter, const char* thing);

void native_greeter_free(NativeGreeter* greeter);

#endif  // EXAMPLES_D_HELLO_LIB_NATIVE_GREETER_H_
