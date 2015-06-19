new_http_archive(
    name = "closure_compiler",
    build_file = "tools/build_rules/closure/closure_compiler.BUILD",
    sha256 = "8e59c1996b8b114c60570f47f36168f111152f4ca9029562e971c987b2aee23a",
    url = "http://dl.google.com/closure-compiler/compiler-20150609.zip",
)

http_jar(
    name = "closure_stylesheets",
    sha256 = "8b2ae8ec3733171ec0c2e6536566df0b3c6da3e59b4784993bc9e73125d29c82",
    url = "https://closure-stylesheets.googlecode.com/files/closure-stylesheets-20111230.jar",
)

new_http_archive(
    name = "closure_templates",
    build_file = "tools/build_rules/closure/closure_templates.BUILD",
    sha256 = "72f87d71e1e9bf297fa6f8d9ec4e615c6e278c8e0ee37ac6e1eb625bb1806440",
    url = "https://closure-templates.googlecode.com/files/closure-templates-for-javascript-2012-12-21.zip",
)

bind(
    name = "closure_compiler_",
    actual = "@closure_compiler//:closure_compiler",
)

bind(
    name = "closure_templates_",
    actual = "@closure_templates//:closure_templates",
)
