def _testproto(ctx):
  for proto in ctx.attr.protos:
    print("proto: ", proto)
    print("  dir(proto): ", dir(proto))
    print("  dir(proto.proto): ", dir(proto.proto))
  for java in ctx.attr.javas:
    print("java : ", java)
    print("  dir(java): ", dir(java))
    print("  java[JavaInfo]: ", java[JavaInfo])
    # print("  dir(proto): ", dir(java.proto))
  print("x:", OutputGroupInfo)
  # print("ProtoInfo: ", ProtoInfo)
  fail("TODO")


testproto = rule(_testproto,
                 attrs = {
                     "protos": attr.label_list(),
                     "javas": attr.label_list(),
                     
                     })
