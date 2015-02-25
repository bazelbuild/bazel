package(default_visibility = ["//visibility:public"])

extra_action(
    name = "ea",
    cmd = "$(location :sh) --extra_action_info=$(EXTRA_ACTION_FILE)",
    tools = [":sh"],
)

action_listener(
    name = "al",
    extra_actions = [":ea"],
    mnemonics = ["CppCompile"],
)

sh_binary(
    name = "sh",
    srcs = ["sh.sh"],
)
