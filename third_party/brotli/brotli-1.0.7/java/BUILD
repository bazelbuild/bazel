package(
    default_visibility = ["//visibility:public"],
)

# >>> JNI headers

genrule(
    name = "copy_link_jni_header",
    srcs = ["@openjdk_jni_h//file"],
    outs = ["jni/jni.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_link_jni_md_header",
    srcs = select({
        "@org_brotli//:darwin": ["@openjdk_macosx_jni_md_h//file"],
        "@org_brotli//:darwin_x86_64": ["@openjdk_macosx_jni_md_h//file"],
        "@org_brotli//:windows_msys": ["@openjdk_windows_jni_md_h//file"],
        "@org_brotli//:windows_msvc": ["@openjdk_windows_jni_md_h//file"],
        "@org_brotli//:windows": ["@openjdk_windows_jni_md_h//file"],
        "//conditions:default": ["@openjdk_solaris_jni_md_h//file"],
    }),
    outs = ["jni/jni_md.h"],
    cmd = "cp -f $< $@",
)

cc_library(
    name = "jni_inc",
    hdrs = [
        ":jni/jni.h",
        ":jni/jni_md.h",
    ],
    includes = ["jni"],
)

# <<< JNI headers

########################################################
# WARNING: do not (transitively) depend on this target!
########################################################
cc_binary(
    name = "brotli_jni.dll",
    srcs = [
        "//org/brotli/wrapper/common:jni_src",
        "//org/brotli/wrapper/dec:jni_src",
        "//org/brotli/wrapper/enc:jni_src",
        "@org_brotli//:common_headers",
        "@org_brotli//:common_sources",
        "@org_brotli//:dec_headers",
        "@org_brotli//:dec_sources",
        "@org_brotli//:enc_headers",
        "@org_brotli//:enc_sources",
    ],
    linkshared = 1,
    deps = [
        ":jni_inc",
        "@org_brotli//:brotli_inc",
    ],
)

########################################################
# WARNING: do not (transitively) depend on this target!
########################################################
cc_binary(
    name = "brotli_jni_no_dictionary_data.dll",
    srcs = [
        "//org/brotli/wrapper/common:jni_src",
        "//org/brotli/wrapper/dec:jni_src",
        "//org/brotli/wrapper/enc:jni_src",
        "@org_brotli//:common_headers",
        "@org_brotli//:common_sources",
        "@org_brotli//:dec_headers",
        "@org_brotli//:dec_sources",
        "@org_brotli//:enc_headers",
        "@org_brotli//:enc_sources",
    ],
    defines = [
        "BROTLI_EXTERNAL_DICTIONARY_DATA=",
    ],
    linkshared = 1,
    deps = [
        ":jni_inc",
        "@org_brotli//:brotli_inc",
    ],
)
