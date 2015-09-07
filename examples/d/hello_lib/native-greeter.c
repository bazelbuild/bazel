// Copyright 2015 Google Inc. All rights reserved.
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

#include "examples/d/hello_lib/native-greeter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

NativeGreeter* native_greeter_new(const char* greeting) {
  if (greeting == NULL) {
    return NULL;
  }
  NativeGreeter* greeter = NULL;
  greeter = (NativeGreeter*)malloc(sizeof(*greeter));
  if (greeter == NULL) {
    return NULL;
  }
  greeter->greeting = strdup(greeting);
  return greeter;
}

void native_greeter_greet(const NativeGreeter* greeter, const char* thing) {
  if (greeter == NULL || thing == NULL) {
    return;
  }
  printf("%s %s!\n", greeter->greeting, thing);
}

void native_greeter_free(NativeGreeter* greeter) {
  if (greeter == NULL) {
    return;
  }
  if (greeter->greeting != NULL) {
    free(greeter->greeting);
  }
  free(greeter);
}
