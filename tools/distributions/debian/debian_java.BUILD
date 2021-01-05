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

# libgoogle-auto-common-java
java_import(
    name = "auto_common",
    jars = ["auto-common.jar"],
)

# libgoogle-auto-service-java
java_import(
    name = "auto_service_lib",
    jars = [
        "auto-service.jar",
        "auto-service-annotations.jar",
    ],
)

# libescapevelocity-java
java_import(
    name = "escapevelocity",
    jars = ["escapevelocity.jar"],
)

# libgoogle-auto-value-java
java_import(
    name = "auto_value_value",
    jars = [
        "auto-value.jar",
        "auto-value-annotations.jar",
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
        "auto-value.jar",
        "auto-value-annotations.jar",
        "auto-common.jar",
        "escapevelocity.jar",
        "javapoet.jar",
    ],
)

# libgoogle-http-client-java
# libgoogle-api-client-java
java_import(
    name = "api_client",
    jars = [
        "google-api-client.jar",
        "google-api-client-jackson2.jar",
        "google-http-client.jar",
        "google-http-client-jackson2.jar",
    ],
    runtime_deps = [
        ":jackson2",
    ],
)

# libgoogle-auth-java
java_import(
    name = "auth",
    jars = [
        "google-auth-library-oauth2-http.jar",
        "google-auth-library-credentials.jar",
    ],
    runtime_deps = [
        ":api_client",
        ":guava",
        "@//third_party/aws-sdk-auth-lite",
    ],
)

# liberror-prone-java
java_import(
    name = "error_prone_annotations",
    jars = [
        "error-prone-annotations.jar",
        "error-prone-type-annotations.jar",
    ],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "error_prone_annotations-jar",
    srcs = ["error-prone-annotations.jar"],
)

# libdiffutils-java
java_import(
    name = "java-diff-utils",
    jars = ["java-diff-utils.jar"],
)

# libopencensus-java
java_import(
    name = "opencensus-api",
    jars = [
        "opencensus-api.jar",
        "opencensus-contrib-grpc-metrics.jar",
    ],
)

# libperfmark-java
java_import(
    name = "perfmark-api",
    jars = [
        "perfmark-api.jar",
    ],
)

# libgoogle-flogger-java
java_import(
    name = "flogger",
    jars = [
        "flogger.jar",
        "flogger-system-backend.jar",
        "google-extensions.jar",
    ],
)

# For bootstrapping JavaBuilder
filegroup(
    name = "flogger-jars",
    srcs = [
        "flogger.jar",
        "flogger-system-backend.jar",
        "google-extensions.jar",
    ],
)

# libchecker-framework-java
java_import(
    name = "checker_framework_annotations",
    jars = ["checker-qual.jar"],
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
        "grpc-api.jar",
        "grpc-auth.jar",
        "grpc-context.jar",
        "grpc-core.jar",
        "grpc-netty.jar",
        "grpc-protobuf.jar",
        "grpc-protobuf-lite.jar",
        "grpc-stub.jar",
    ],
)

# junit4
java_import(
    name = "junit4",
    jars = [
        "hamcrest-core.jar",
        "junit4.jar",
    ],
)
