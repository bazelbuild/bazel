cc_binary(
    name = "libzstd-jni.so",
    srcs = glob([
        "src/main/native/**/*.c",
        "src/main/native/**/*.h",
    ]) + select({
        "@io_bazel//src/conditions:windows": [
            "src/windows/include/jni_md.h",
            "jni/jni.h",
        ],
        "//conditions:default": [
            "jni/jni_md.h",
            "jni/jni.h",
        ]
    }),
    copts = select({
        "@io_bazel//src/conditions:windows": [],
        "//conditions:default": [
            "-std=c99",
            "-Wno-unused-variable",
            "-Wno-sometimes-uninitialized",
        ]
    }),
    linkshared = 1,
    includes = select({
        "@io_bazel//src/conditions:windows": ["src/windows/include"],
        "//conditions:default": [],
    }) + [
        "jni",
        "src/main/native",
        "src/main/native/common",
    ],
    local_defines = [
        "ZSTD_LEGACY_SUPPORT=4",
        "ZSTD_MULTITHREAD=1",
    ] + select({
        "@io_bazel//src/conditions:windows": ["_JNI_IMPLEMENTATION_"],
        "//conditions:default": [],
    }),
)


genrule(
    name = "version-java",
    cmd_bash = 'echo "package com.github.luben.zstd.util;\n\npublic class ZstdVersion {\n\tpublic static final String VERSION = \\"$$(cat $<)\\";\n}" > $@',
    cmd_ps = '$$PSDefaultParameterValues.Remove("*:Encoding"); $$version = (Get-Content $<) -join ""; Set-Content -NoNewline -Path $@ -Value "package com.github.luben.zstd.util;\n\npublic class ZstdVersion {\n\tpublic static final String VERSION = `"$${version}`";\n}\n"',
    srcs = ["version"],
    outs = ["ZstdVersion.java"],
)

java_library(
    name = "zstd-jni",
    srcs = glob([
        "src/main/java/**/*.java",
    ]) + [
        ":version-java",
    ],
    resources = [":libzstd-jni.so"],
    visibility = [
        "//visibility:public",
    ],
)
