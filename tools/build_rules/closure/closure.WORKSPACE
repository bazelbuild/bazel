new_http_archive(
    name = "closure_compiler",
    build_file = "tools/build_rules/closure/closure_compiler.BUILD",
    sha256 = "e4e0cb49ad21ec26dd47693bdbd48f67aefe2d94fe8d9239312d2bcc74986538",
    url = "http://dl.google.com/closure-compiler/compiler-20150729.zip",
)

new_git_repository(
    name = "closure_library",
    build_file = "tools/build_rules/closure/closure_library.BUILD",
    commit = "748b32441093c1474db2e0b3d074250e0bc47778",
    remote = "https://github.com/google/closure-library.git",
)

http_jar(
    name = "closure_stylesheets",
    sha256 = "8b2ae8ec3733171ec0c2e6536566df0b3c6da3e59b4784993bc9e73125d29c82",
    url = "https://closure-stylesheets.googlecode.com/files/closure-stylesheets-20111230.jar",
)

new_http_archive(
    name = "closure_templates",
    build_file = "tools/build_rules/closure/closure_templates.BUILD",
    sha256 = "cdd94123cd0d1c3a183c15e855739c0aa5390297c22dddc731b8d7b23815e8a2",
    url = "http://dl.google.com/closure-templates/closure-templates-for-javascript-latest.zip",
)

bind(
    name = "closure_compiler_",
    actual = "@closure_compiler//:closure_compiler",
)

bind(
    name = "closure_templates_",
    actual = "@closure_templates//:closure_templates",
)
