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
        "java/netty-buffer.jar",
        "java/netty-codec-dns.jar",
        "java/netty-codec-haproxy.jar",
        "java/netty-codec-http2.jar",
        "java/netty-codec-http.jar",
        "java/netty-codec.jar",
        "java/netty-codec-memcache.jar",
        "java/netty-codec-mqtt.jar",
        "java/netty-codec-redis.jar",
        "java/netty-codec-smtp.jar",
        "java/netty-codec-socks.jar",
        "java/netty-codec-stomp.jar",
        "java/netty-common.jar",
        "java/netty-handler.jar",
        "java/netty-handler-proxy.jar",
        "java/netty-resolver-dns.jar",
        "java/netty-resolver-dns-native-macos.jar",
        "java/netty-resolver.jar",
        "java/netty-transport.jar",
        "java/netty-transport-native-epoll.jar",
        "java/netty-transport-native-kqueue.jar",
        "java/netty-transport-native-unix-common.jar",
    ],
)

# libgoogle-gson-java
java_import(
    name = "gson",
    jars = ["java/gson.jar"],
)

# libtomcat9-java
java_import(
    name = "tomcat_annotations_api",
    jars = ["java/tomcat9-annotations-api.jar"],
)

# For bootstrapping java toolcahin
filegroup(
    name = "tomcat_annotations_api-jars",
    srcs = ["java/tomcat9-annotations-api.jar"],
)

# libjava-allocation-instrumenter-java
java_import(
    name = "allocation_instrumenter",
    jars = ["java/java-allocation-instrumenter.jar"],
)

# libprotobuf-java
java_import(
    name = "protobuf_java",
    jars = ["java/protobuf.jar"],
)

# libprotobuf-java
java_import(
    name = "protobuf_java_util",
    jars = ["java/protobuf-util.jar"],
)

# For bootstrapping java toolcahin
filegroup(
    name = "bootstrap-derived-java-jars",
    srcs = [
        "java/protobuf.jar",
        "java/protobuf-util.jar",
    ],
)

# libcommons-collections3-java
java_import(
    name = "apache_commons_collections",
    jars = ["java/commons-collections3.jar"],
)

# libcommons-lang-java
java_import(
    name = "apache_commons_lang",
    jars = ["java/commons-lang.jar"],
)

# libcommons-compress-java
java_import(
    name = "apache_commons_compress",
    jars = ["java/commons-compress.jar"],
)

# libcommons-pool2-java
java_import(
    name = "apache_commons_pool2",
    jars = ["java/commons-pool2.jar"],
)

# velocity
java_import(
    name = "apache_velocity",
    jars = ["java/velocity.jar"],
)

# libasm-java
java_import(
    name = "asm",
    jars = ["java/asm.jar"],
)

# libjackson2-core-java
java_import(
    name = "jackson2",
    jars = [
        "java/jackson-core.jar",
    ],
)

# libjcip-annotations-java
java_import(
    name = "jcip_annotations",
    jars = [
        "java/jcip-annotations.jar",
    ],
)

# For bootstrapping java toolcahin
filegroup(
    name = "jcip_annotations-jars",
    srcs = [
        "java/jcip-annotations.jar",
    ],
)

# libjsr305-java
java_import(
    name = "jsr305",
    jars = ["java/jsr305.jar"],
)

# For bootstrapping java toolcahin
filegroup(
    name = "jsr305-jars",
    srcs = ["java/jsr305.jar"],
)

# libnetty-tcnative-java
java_import(
    name = "netty_tcnative",
    jars = ["java/netty-tcnative.jar"],
)

# libjavapoet-java
java_import(
    name = "javapoet",
    jars = ["java/javapoet.jar"],
)

# libjaxb-api-java
java_import(
    name = "jaxb",
    jars = ["java/jaxb-api.jar"],
)

# libxz-java
java_import(
    name = "xz",
    jars = ["java/xz.jar"],
)

# libgeronimo-annotation-1.3-spec-java
java_import(
    name = "javax_annotations",
    jars = ["java/geronimo-annotation-1.3-spec.jar"],
    neverlink = 1,
)

# libandroid-tools-common-java
# libandroid-tools-repository-java
# libandroid-layoutlib-api-java
java_import(
    name = "android_common_25_0_0_lite",
    jars = [
        "java/com.android.tools.common.jar",
        "java/com.android.tools.repository.jar",
        "java/com.android.tools.layoutlib.layoutlib-api.jar",
    ],
)

# libguava-java
java_import(
    name = "guava",
    jars = ["java/guava.jar"],
    exports = [
        "@//third_party:error_prone_annotations",
        "@//third_party:jcip_annotations",
        "@//third_party:jsr305",
    ],
)

# For bootstrapping java toolcahin
filegroup(
    name = "guava-jars",
    srcs = ["java/guava.jar"],
)

# libjacoco-java - BEGIN
JACOCOVERSION = "0.8.6"

java_import(
    name = "agent",
    jars = ["java/org.jacoco.agent.jar"],
)

java_import(
    name = "agent-%s" % JACOCOVERSION,
    jars = ["java/org.jacoco.agent.jar"],
)

java_import(
    name = "core",
    jars = ["java/org.jacoco.core.jar"],
)

java_import(
    name = "core-%s" % JACOCOVERSION,
    jars = ["java/org.jacoco.core.jar"],
)

filegroup(
    name = "core-jars",
    srcs = ["java/org.jacoco.core.jar"],
)

filegroup(
    name = "core-jars-%s" % JACOCOVERSION,
    srcs = ["java/org.jacoco.core.jar"],
)

java_import(
    name = "report",
    jars = ["java/org.jacoco.report.jar"],
)

java_import(
    name = "report-%s" % JACOCOVERSION,
    jars = ["java/org.jacoco.report.jar"],
)

java_import(
    name = "blaze-agent",
    jars = ["java/org.jacoco.agent.jar"],
)

java_import(
    name = "blaze-agent-%s" % JACOCOVERSION,
    jars = ["java/org.jacoco.agent.jar"],
)
# libjacoco-java - END

# libgoogle-auto-common-java
java_import(
    name = "auto_common",
    jars = ["java/auto-common.jar"],
)

# libgoogle-auto-service-java
java_import(
    name = "auto_service_lib",
    jars = [
        "java/auto-service.jar",
        "java/auto-service-annotations.jar",
    ],
)

# libescapevelocity-java
java_import(
    name = "escapevelocity",
    jars = ["java/escapevelocity.jar"],
)

# libgoogle-auto-value-java
java_import(
    name = "auto_value_value",
    jars = [
        "java/auto-value.jar",
        "java/auto-value-annotations.jar",
    ],
    runtime_deps = [
        ":escapevelocity",
        ":javapoet",
    ],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "auto_value-jars",
    srcs = [
        "java/auto-common.jar",
        "java/auto-value.jar",
        "java/auto-value-annotations.jar",
        "java/escapevelocity.jar",
        "java/javapoet.jar",
    ],
)

# libgoogle-http-client-java
# libgoogle-api-client-java
java_import(
    name = "api_client",
    jars = [
        "java/google-api-client.jar",
        "java/google-api-client-jackson2.jar",
        "java/google-http-client.jar",
        "java/google-http-client-jackson2.jar",
    ],
    runtime_deps = [
        ":jackson2",
    ],
)

# libgoogle-auth-java
java_import(
    name = "auth",
    jars = [
        "java/google-auth-library-oauth2-http.jar",
        "java/google-auth-library-credentials.jar",
    ],
    runtime_deps = [
        ":api_client",
        ":guava",
    ],
)

# liberror-prone-java
java_import(
    name = "error_prone_annotations",
    jars = [
        "java/error-prone-annotations.jar",
        "java/error-prone-type-annotations.jar",
    ],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "error_prone_annotations-jar",
    srcs = ["java/error-prone-annotations.jar"],
)

# libdiffutils-java
java_import(
    name = "java-diff-utils",
    jars = ["java/java-diff-utils.jar"],
)

# libopencensus-java
java_import(
    name = "opencensus-api",
    jars = [
        "java/opencensus-api.jar",
        "java/opencensus-contrib-grpc-metrics.jar",
    ],
)

# libperfmark-java
java_import(
    name = "perfmark-api",
    jars = [
        "java/perfmark-api.jar",
    ],
)

# libgoogle-flogger-java
java_import(
    name = "flogger",
    jars = [
        "java/flogger.jar",
        "java/flogger-system-backend.jar",
        "java/google-extensions.jar",
    ],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "flogger-jars",
    srcs = [
        "java/flogger.jar",
        "java/flogger-system-backend.jar",
        "java/google-extensions.jar",
    ],
)

# libchecker-framework-java
java_import(
    name = "checker_framework_annotations",
    jars = ["java/checker-qual.jar"],
)

# libgrpc-java
java_import(
    name = "grpc-jar",
    jars = [":bootstrap-grpc-jars"],
    runtime_deps = [
        ":netty",
        ":opencensus-api",
        ":perfmark-api",
    ],
    deps = [
        ":guava",
    ],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "bootstrap-grpc-jars",
    srcs = [
        "java/grpc-api.jar",
        "java/grpc-auth.jar",
        "java/grpc-context.jar",
        "java/grpc-core.jar",
        "java/grpc-netty.jar",
        "java/grpc-protobuf.jar",
        "java/grpc-protobuf-lite.jar",
        "java/grpc-stub.jar",
    ],
)

# junit4
java_import(
    name = "junit4",
    jars = [
        "java/hamcrest-core.jar",
        "java/junit4.jar",
    ],
)

# libreactive-streams-java
java_import(
    name = "reactive_streams",
    jars = ["java/reactive-streams.jar"],
)

# librx-java
java_import(
    name = "rxjava3",
    jars = ["java/rxjava.jar"],
    deps = [":reactive_streams"],
)

# libcaffeine-java
java_import(
    name = "caffeine",
    jars = ["java/caffeine.jar"],
)
