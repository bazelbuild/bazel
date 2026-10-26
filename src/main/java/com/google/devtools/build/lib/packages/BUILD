load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/main/java/com/google/devtools/build/lib:build_lib"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "configured_attribute_mapper",
    srcs = ["ConfiguredAttributeMapper.java"],
    deps = [
        ":packages",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:core_options",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect:collection_utils",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "globber",
    srcs = ["Globber.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/main/java/com/google/devtools/build/lib:package_loader_access",
        "//src/main/java/com/google/devtools/build/lib/skyframe:__pkg__",
    ],
)

java_library(
    name = "globber_utils",
    srcs = ["GlobberUtils.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/main/java/com/google/devtools/build/lib:package_loader_access",
        "//src/main/java/com/google/devtools/build/lib/skyframe:__pkg__",
    ],
    deps = [
        ":globber",
        "//third_party/java/error_prone:annotations",
    ],
)

java_library(
    name = "packages",
    srcs = glob(
        ["*.java"],
        exclude = [
            "AdvertisedProviderSet.java",
            "AspectParameters.java",
            "BuilderFactoryForTesting.java",  # see builder_factory_for_testing
            "BuildFileName.java",
            "BuildFileNotFoundException.java",
            "BuiltinProvider.java",
            "BzlInitThreadContext.java",
            "BzlVisibility.java",
            "ConfigurationFragmentPolicy.java",
            "ConfiguredAttributeMapper.java",
            "DeclaredExecGroup.java",
            "Globber.java",
            "GlobberUtils.java",
            "Info.java",
            "InvalidPackageNameException.java",
            "LabelConverter.java",
            "LabelPrinter.java",
            "License.java",
            "NativeInfo.java",
            "NoSuchPackageException.java",
            "NoSuchPackagePieceException.java",
            "NoSuchThingException.java",
            "PackageGroupsRuleVisibility.java",
            "PackagePieceIdentifier.java",
            "PackageSpecification.java",
            "Provider.java",
            "RepoThreadContext.java",
            "RepositoryFetchException.java",
            "RequiredProviders.java",
            "RuleClassUtils.java",
            "RuleVisibility.java",
            "StarlarkExportable.java",
            "StarlarkInfo.java",
            "StarlarkInfoNoSchema.java",
            "StarlarkInfoWithMessage.java",
            "StarlarkInfoWithSchema.java",
            "StarlarkProvider.java",
            "StarlarkProviderIdentifier.java",
            "StructImpl.java",
            "StructProvider.java",
            "TriState.java",
            "VendorThreadContext.java",
            "AspectPropagationPredicate.java",
            "BuiltinRestriction.java",
            "BuiltinsInternalModule.java",
            "CachingPackageLocator.java",
            "DotBazelFileSyntaxChecker.java",
            "EnumFilterConverter.java",
            "FunctionSplitTransitionAllowlist.java",
            "PredicateWithMessage.java",
            "Proto.java",
            "RelativePackageNameResolver.java",
            "RepoFileGlobals.java",
            "TargetSuggester.java",
            "VendorFileGlobals.java",
        ],
    ),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/main/java/com/google/devtools/build/lib:package_loader_access",
    ],
    exports = [
        ":advertised_provider_set",
        ":aspect_parameters",
        ":aspect_propagation_predicate",
        ":build_file_name",
        ":builtin_restriction",
        ":builtins_internal_module",
        ":caching_package_locator",
        ":configuration_fragment_policy",
        ":dot_bazel_file_syntax_checker",
        ":enum_filter_converter",
        ":function_split_transition_allowlist",
        ":label_converter",
        ":license",
        ":no_such_package_exception",
        ":package_loading_exceptions",
        ":predicate_with_message",
        ":relative_package_name_resolver",
        ":repo_file_globals",
        ":required_providers",
        ":target_suggester",
        ":thread_contexts",
        ":tristate",
    ],
    deps = [
        ":advertised_provider_set",
        ":aspect_parameters",
        ":aspect_propagation_predicate",
        ":build_file_name",
        ":builtin_restriction",
        ":builtins_internal_module",
        ":bzl_visibility",
        ":caching_package_locator",
        ":configuration_fragment_policy",
        ":declared_exec_group",
        ":dot_bazel_file_syntax_checker",
        ":enum_filter_converter",
        ":function_split_transition_allowlist",
        ":globber",
        ":globber_utils",
        ":label_converter",
        ":label_printer",
        ":license",
        ":no_such_package_exception",
        ":package_loading_exceptions",
        ":package_piece_identifier",
        ":package_specification",
        ":predicate_with_message",
        ":provider",
        ":relative_package_name_resolver",
        ":repo_file_globals",
        ":required_providers",
        ":rule_visibility",
        ":starlark_exportable",
        ":target_suggester",
        ":thread_contexts",
        ":tristate",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/actions:thread_state_receiver",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:feature_set",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_class_set",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:toolchain_type_requirement",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:no_transition",
        "//src/main/java/com/google/devtools/build/lib/analysis/config/transitions:transition_factory",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:constants",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/collect:collection_utils",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:nested_sets_should_be_interned_by_equality",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/io:file_symlink_exception",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/pkgcache:package_options",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:abstract-exported-starlark-symbol-codec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/config",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:hash_codes",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/lib",
        "//src/main/java/net/starlark/java/spelling",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:build_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/auto:auto_value",
        "//third_party/java/caffeine",
        "//third_party/java/flogger",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/guava:escape",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf",
        "//third_party/java/protobuf:protobuf_lite",
    ],
)

java_library(
    name = "bzl_visibility",
    srcs = ["BzlVisibility.java"],
    deps = [
        ":package_specification",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "declared_exec_group",
    srcs = ["DeclaredExecGroup.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:toolchain_type_requirement",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "package_specification",
    srcs = ["PackageSpecification.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:nested_sets_should_be_interned_by_equality",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "builder_factory_for_testing",
    srcs = ["BuilderFactoryForTesting.java"],
    deps = [
        ":packages",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//third_party/java/guava:annotations",
    ],
)

java_library(
    name = "label_printer",
    srcs = ["LabelPrinter.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "package_piece_identifier",
    srcs = ["PackagePieceIdentifier.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/util:hash_codes",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party/java/guava:base",
    ],
)

java_library(
    name = "provider",
    srcs = [
        "BuiltinProvider.java",
        "Info.java",
        "NativeInfo.java",
        "Provider.java",
        "StarlarkInfo.java",
        "StarlarkInfoNoSchema.java",
        "StarlarkInfoWithMessage.java",
        "StarlarkInfoWithSchema.java",
        "StarlarkProvider.java",
        "StarlarkProviderIdentifier.java",
        "StructImpl.java",
        "StructProvider.java",
    ],
    deps = [
        ":starlark_exportable",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:starlark_builtins_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:abstract-exported-starlark-symbol-codec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:hash_codes",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/lib",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "rule_class_id",
    srcs = ["RuleClassId.java"],
    deps = ["//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec"],
)

java_library(
    name = "rule_class_utils",
    srcs = ["RuleClassUtils.java"],
    deps = [
        ":packages",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/skyframe:starlark_builtins_value",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "rule_visibility",
    srcs = [
        "PackageGroupsRuleVisibility.java",
        "RuleVisibility.java",
    ],
    deps = [
        ":package_specification",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/auto:auto_value",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "starlark_exportable",
    srcs = ["StarlarkExportable.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
    ],
)

java_library(
    name = "license",
    srcs = ["License.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "tristate",
    srcs = ["TriState.java"],
)

java_library(
    name = "label_converter",
    srcs = ["LabelConverter.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "no_such_package_exception",
    srcs = [
        "NoSuchPackageException.java",
        "NoSuchThingException.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/java/guava:base",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "advertised_provider_set",
    srcs = ["AdvertisedProviderSet.java"],
    deps = [
        ":provider",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "package_loading_exceptions",
    srcs = [
        "BuildFileNotFoundException.java",
        "InvalidPackageNameException.java",
        "NoSuchPackagePieceException.java",
        "RepositoryFetchException.java",
    ],
    deps = [
        ":no_such_package_exception",
        ":package_piece_identifier",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
    ],
)

java_library(
    name = "aspect_parameters",
    srcs = ["AspectParameters.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:serialization-constant",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:collect",
        "//third_party/java/protobuf",
    ],
)

java_library(
    name = "thread_contexts",
    srcs = [
        "BzlInitThreadContext.java",
        "RepoThreadContext.java",
        "VendorThreadContext.java",
    ],
    deps = [
        ":bzl_visibility",
        "//src/main/java/com/google/devtools/build/lib/analysis:rule_definition_environment",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "configuration_fragment_policy",
    srcs = ["ConfigurationFragmentPolicy.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment_class_set",
        "//src/main/java/net/starlark/java/annot",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "required_providers",
    srcs = ["RequiredProviders.java"],
    deps = [
        ":advertised_provider_set",
        ":provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:transitive_info_provider",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:visible-for-serialization",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "build_file_name",
    srcs = ["BuildFileName.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
    ],
)

java_library(
    name = "caching_package_locator",
    srcs = ["CachingPackageLocator.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "dot_bazel_file_syntax_checker",
    srcs = ["DotBazelFileSyntaxChecker.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "enum_filter_converter",
    srcs = ["EnumFilterConverter.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "predicate_with_message",
    srcs = ["PredicateWithMessage.java"],
    deps = [
        "//third_party/java/guava:base",
    ],
)

java_library(
    name = "relative_package_name_resolver",
    srcs = ["RelativePackageNameResolver.java"],
    deps = [
        ":package_loading_exceptions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party/java/guava:base",
    ],
)

java_library(
    name = "target_suggester",
    srcs = ["TargetSuggester.java"],
    deps = [
        "//src/main/java/net/starlark/java/spelling",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
    ],
)

java_library(
    name = "repo_file_globals",
    srcs = [
        "RepoFileGlobals.java",
        "VendorFileGlobals.java",
    ],
    deps = [
        ":thread_contexts",
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "builtin_restriction",
    srcs = [
        "BuiltinRestriction.java",
        "Proto.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/lib",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "aspect_propagation_predicate",
    srcs = ["AspectPropagationPredicate.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "builtins_internal_module",
    srcs = ["BuiltinsInternalModule.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/docgen/annot",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "function_split_transition_allowlist",
    srcs = ["FunctionSplitTransitionAllowlist.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
    ],
)
