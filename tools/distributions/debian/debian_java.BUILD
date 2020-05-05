# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//visibility:public"])

# libnetty-java
# TODO: The netty-all.jar is empty in Debian distribution, we have to list
# all netty jars here. Replace them with netty-all.jar once it's fixed.
java_import(
    name = "netty",
    jars = [
        "netty-buffer.jar",
        "netty-codec-dns.jar",
        "netty-codec-haproxy.jar",
        "netty-codec-http2.jar",
        "netty-codec-http.jar",
        "netty-codec.jar",
        "netty-codec-memcache.jar",
        "netty-codec-mqtt.jar",
        "netty-codec-redis.jar",
        "netty-codec-smtp.jar",
        "netty-codec-socks.jar",
        "netty-codec-stomp.jar",
        "netty-common.jar",
        "netty-handler.jar",
        "netty-handler-proxy.jar",
        "netty-resolver-dns.jar",
        "netty-resolver-dns-native-macos.jar",
        "netty-resolver.jar",
        "netty-transport.jar",
        "netty-transport-native-epoll.jar",
        "netty-transport-native-kqueue.jar",
        "netty-transport-native-unix-common.jar",
        "netty-transport-sctp.jar",
    ],
)

# libgoogle-gson-java
java_import(
    name = "gson",
    jars = ["gson.jar"],
)

# libtomcat9-java
java_import(
    name = "tomcat_annotations_api",
    jars = ["tomcat9-annotations-api.jar"],
)

# libjava-allocation-instrumenter-java
java_import(
    name = "allocation_instrumenter",
    jars = ["java-allocation-instrumenter.jar"],
)
