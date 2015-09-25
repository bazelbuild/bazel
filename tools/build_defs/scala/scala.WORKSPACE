new_http_archive(
  name = "scala-archive",
  url = "http://downloads.typesafe.com/scala/2.11.6/scala-2.11.6.tgz",
  sha256 = "41ba45e4600404634217a66d6b2c960459d3a67e0344a7c3d9642d0eaa446583",
  build_file = "tools/build_defs/scala/scala.BUILD",
  type = "tgz",
)

bind(
  name = "scala-lib",
  actual = "@scala-archive//:lib/scala-library.jar",
)

bind(
  name = "scalac",
  actual = "@scala-archive//:bin/scalac",
)
