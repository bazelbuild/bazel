inline fun freeInlineFun(op: () -> Unit) { op() }

class ClassWithInline {
    inline fun classInlineFun(op: () -> Unit) { op() }
}

object ObjectWithInline {
    inline fun objectInlineFun(op: () -> Unit) { op() }
}

abstract class AbstractClassWithInline {
    inline fun inheritedInlineFun(op: () -> Unit) { op() }
}

object ObjectInheritingInline: AbstractClassWithInline()
