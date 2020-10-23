"""The input file for struct default values test"""

def check_struct_default_values(
        struct_no_args = struct(),
        struct_arg = struct(foo = "bar"),
        struct_args = struct(foo = "bar", bar = "foo"),
        struct_int_args = struct(one = 1, two = 2, three = 3),
        struct_struct_args = struct(
            none = struct(),
            one = struct(foo = "bar"),
            multiple = struct(one = 1, two = 2, three = 3),
        )):
    """Checks the default values of structs.

    Args:
        struct_no_args: struct with no arguments
        struct_arg: struct with one argument
        struct_args: struct with multiple arguments
        struct_int_args: struct with int arguments
        struct_struct_args: struct with struct arguments
    """
    pass
