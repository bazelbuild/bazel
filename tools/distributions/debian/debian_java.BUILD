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

# For bootstrapping java toolcahin
filegroup(
    name = "tomcat_annotations_api-jars",
    srcs = ["tomcat9-annotations-api.jar"],
)

# libjava-allocation-instrumenter-java
java_import(
    name = "allocation_instrumenter",
    jars = ["java-allocation-instrumenter.jar"],
)

# libprotobuf-java
java_import(
    name = "protobuf_java",
    jars = ["protobuf.jar"],
)

# libprotobuf-java
java_import(
    name = "protobuf_java_util",
    jars = ["protobuf-util.jar"],
)

# For bootstrapping java toolcahin
filegroup(
    name = "bootstrap-derived-java-jars",
    srcs = [
        "protobuf.jar",
        "protobuf-util.jar",
    ],
)

# libcommons-collections3-java
java_import(
    name = "apache_commons_collections",
    jars = ["commons-collections3.jar"],
)

# libcommons-lang-java
java_import(
    name = "apache_commons_lang",
    jars = ["commons-lang.jar"],
)

# libcommons-compress-java
java_import(
    name = "apache_commons_compress",
    jars = ["commons-compress.jar"],
)

# libcommons-logging-java
java_import(
    name = "apache_commons_logging",
    jars = ["commons-logging.jar"],
)

# libcommons-pool2-java
java_import(
    name = "apache_commons_pool2",
    jars = ["commons-pool2.jar"],
)

# velocity
java_import(
    name = "apache_velocity",
    jars = ["velocity.jar"],
)

# libasm-java
java_import(
    name = "asm",
    jars = ["asm.jar"],
)

# libjackson2-core-java
java_import(
    name = "jackson2",
    jars = [
        "jackson-core.jar",
    ],
)

# libjcip-annotations-java
java_import(
    name = "jcip_annotations",
    jars = [
        "jcip-annotations.jar",
    ],
)

# For bootstrapping java toolcahin
filegroup(
    name = "jcip_annotations-jars",
    srcs = [
        "jcip-annotations.jar",
    ],
)

# libjsr305-java
java_import(
    name = "jsr305",
    jars = ["jsr305.jar"],
)

# For bootstrapping java toolcahin
filegroup(
    name = "jsr305-jars",
    srcs = ["jsr305.jar"],
)

# libnetty-tcnative-java
java_import(
    name = "netty_tcnative",
    jars = ["netty-tcnative.jar"],
)

# libjavapoet-java
java_import(
    name = "javapoet",
    jars = ["javapoet.jar"],
)

# libjaxb-api-java
java_import(
    name = "jaxb",
    jars = ["jaxb-api.jar"],
)

# libxz-java
java_import(
    name = "xz",
    jars = ["xz.jar"],
)

# libgeronimo-annotation-1.3-spec-java
java_import(
    name = "javax_annotations",
    jars = ["geronimo-annotation-1.3-spec.jar"],
    neverlink = 1,
)

# libandroid-tools-common-java
# libandroid-tools-repository-java
# libandroid-layoutlib-api-java
java_import(
    name = "android_common_25_0_0_lite",
    jars = [
        "com.android.tools.common.jar",
        "com.android.tools.repository.jar",
        "com.android.tools.layoutlib.layoutlib-api.jar",
    ],
)

# libguava-java
java_import(
    name = "guava",
    jars = ["guava.jar"],
    exports = [
        "@//third_party:error_prone_annotations",
        "@//third_party:jcip_annotations",
        "@//third_party:jsr305",
    ],
)

# For bootstrapping java toolcahin
filegroup(
    name = "guava-jars",
    srcs = ["guava.jar"],
)

# libjacoco-java - BEGIN
JACOCOVERSION = "0.8.3"
java_import(
    name = "agent",
    jars = ["org.jacoco.agent.jar"],
)

java_import(
    name = "agent-%s" % JACOCOVERSION,
    jars = ["org.jacoco.agent.jar"],
)

java_import(
    name = "core",
    jars = ["org.jacoco.core.jar"],
)

java_import(
    name = "core-%s" % JACOCOVERSION,
    jars = ["org.jacoco.core.jar"],
)

filegroup(
    name = "core-jars",
    srcs = ["org.jacoco.core.jar"],
)

filegroup(
    name = "core-jars-%s" % JACOCOVERSION,
    srcs = ["org.jacoco.core.jar"],
)

java_import(
    name = "report",
    jars = ["org.jacoco.report.jar"],
)

java_import(
    name = "report-%s" % JACOCOVERSION,
    jars = ["org.jacoco.report.jar"],
)

java_import(
   name = "blaze-agent",
   jars = ["org.jacoco.agent.jar"],
)

java_import(
   name = "blaze-agent-%s" % JACOCOVERSION,
   jars = ["org.jacoco.agent.jar"],
)
# libjacoco-java - END
