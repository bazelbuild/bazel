// Copyright 2022 The Bazel Authors. All rights reserved.
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
#include <jni.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <net/route.h>
#include <sys/sysctl.h>
#include <sys/types.h>

#include <string>
#include <vector>

#include "src/main/native/unix_jni.h"

namespace blaze_jni {

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_profiler_SystemNetworkStats_getNetIoCountersNative(
    JNIEnv *env, jclass clazz, jobject counters_map) {
  int mib[6] = {CTL_NET,         // networking subsystem
                PF_ROUTE,        // type of information
                0,               // always 0 for CTL_NET/PF_ROUTE
                0,               // 0 = all address family
                NET_RT_IFLIST2,  // operation
                0};

  size_t buf_len;
  if (sysctl(mib, 6, nullptr, &buf_len, nullptr, 0) < 0) {
    PostException(env, errno, "sysctl");
    return;
  }

  std::vector<char> buf(buf_len);
  if (sysctl(mib, 6, buf.data(), &buf_len, nullptr, 0) < 0) {
    PostException(env, errno, "sysctl");
    return;
  }

  jclass map_class = env->GetObjectClass(counters_map);
  jmethodID map_put = env->GetMethodID(
      map_class, "put",
      "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

  jclass counter_class = env->FindClass(
      "com/google/devtools/build/lib/profiler/SystemNetworkStats$NetIoCounter");
  jmethodID counter_create =
      env->GetStaticMethodID(counter_class, "create",
                             "(JJJJ)Lcom/google/devtools/build/lib/profiler/"
                             "SystemNetworkStats$NetIoCounter;");

  const char *end = buf.data() + buf_len;
  for (const char *next = buf.data(); next < end;) {
    if_msghdr *ifm = (if_msghdr *)next;
    next += ifm->ifm_msglen;
    if (ifm->ifm_type != RTM_IFINFO2) {
      continue;
    }

    if_msghdr2 *if2m = (if_msghdr2 *)ifm;

    sockaddr_dl *sdl = (sockaddr_dl *)(if2m + 1);
    std::string sdl_name(sdl->sdl_data, sdl->sdl_nlen);

    jstring name = env->NewStringUTF(sdl_name.c_str());
    jobject counter = env->CallStaticObjectMethod(
        counter_class, counter_create, if2m->ifm_data.ifi_obytes,
        if2m->ifm_data.ifi_ibytes, if2m->ifm_data.ifi_opackets,
        if2m->ifm_data.ifi_ipackets);

    env->CallObjectMethod(counters_map, map_put, name, counter);
  }
}

}  // namespace blaze_jni
