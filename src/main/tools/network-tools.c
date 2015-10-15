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

#define _GNU_SOURCE

#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "process-tools.h"
#include "network-tools.h"

void BringupInterface(const char *name) {
  int fd;

  struct ifreq ifr;

  CHECK_CALL(fd = socket(AF_INET, SOCK_DGRAM, 0));

  memset(&ifr, 0, sizeof(ifr));
  strncpy(ifr.ifr_name, name, IF_NAMESIZE);

  // Verify that name is valid.
  CHECK_CALL(if_nametoindex(ifr.ifr_name));

  // Enable the interface
  ifr.ifr_flags |= IFF_UP;
  CHECK_CALL(ioctl(fd, SIOCSIFFLAGS, &ifr));

  CHECK_CALL(close(fd));
}
